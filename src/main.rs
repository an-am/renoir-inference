mod model;

use burn::tensor::*;
use futures::StreamExt;
use model::deep_model::Model;
use renoir::operator::source::ChannelSource;
use renoir::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{
    postgres::{PgListener, PgPool, PgRow},
    FromRow, Pool, Postgres, Row,
};

use std::{
    sync::{atomic::AtomicUsize, OnceLock},
    time::{Duration, Instant},
};

const MAX_REQUEST: i32 = 5;
const FITTED_LAMBDA_INCOME: f32 = 0.3026418664067109;
const FITTED_LAMBDA_WEALTH: f32 = 0.1336735055366279;
const SCALER_MEANS: [f32; 9] = [
    55.2534,
    0.492,
    2.5106,
    0.41912250425,
    7.664003606794818,
    5.650795740523163,
    0.3836,
    0.5132,
    1.7884450179129336,
];
const SCALER_SCALES: [f32; 9] = [
    11.970496582849016,
    0.49993599590347565,
    0.7617661320904205,
    0.15136821202756753,
    2.4818937424620633,
    1.5813522545815777,
    0.4862623160393987,
    0.49982572962983807,
    0.8569630982206199,
];

type Backend = burn::backend::NdArray<f32>;
type Device = burn::backend::ndarray::NdArrayDevice;

#[derive(Clone, Deserialize, Serialize, Debug, FromRow)]
struct Client {
    row_id: i32,
    age: i32,
    gender: i32,
    family_members: i32,
    financial_education: f32,
    income: f32,
    wealth: f32,
    income_investment: i32,
    accumulation_investment: i32,
    client_id: i32,
}

struct Product {
    row_id: i32,
    income_investment: i32,
    accumulation_investment: i32,
    financial_status: f32,
}

async fn update_needs_get_products(prediction: f32, product: Product, pool: &PgPool) -> (i32, f32) {
    // Connect to the DB
    // println!("Model output for id {}: {}", product.row_id, prediction);

    //  &[&prediction, &product.financial_status, &product.row_id]
    //let mut conn = PgClient::connect(database_url, NoTls).unwrap();
    sqlx::query(
        r#"
        UPDATE needs
        SET risk_propensity = $1, financial_status = $2
        WHERE id = $3
        "#,
    )
    .bind(prediction)
    .bind(product.financial_status)
    .bind(product.row_id)
    .execute(pool)
    .await
    .unwrap();

    tracing::debug!("Querying UPDATE {}", product.row_id);

    // Retrieve products from product table
    if false {
        // SKIP BECAUSE I DON'T HAVE THE TABLE, TODO ADD IT BACK
        let products: Vec<PgRow> = sqlx::query(
            r#"
        SELECT id_product, description 
            FROM products
            WHERE   
                (
                    income = $1
                    OR accumulation = $2
                )
                AND risk <= $3
        "#,
        )
        .bind(product.income_investment)
        .bind(product.accumulation_investment)
        .bind(prediction)
        .fetch_all(pool)
        .await
        .unwrap();

        tracing::debug!(
            "Advised {} products for id={}",
            products.len(),
            product.row_id
        );
    }

    /* for p in products {
        let id_product: i32 = p.get("id_product");
        let description: &str = p.get("description");

        println!("ID: {}, description: {}", id_product, description);
    } */
    (product.row_id, prediction)
}

async fn get_client(id: i32, pool: &PgPool) -> Client {
    // Retrieve the row through its id
    let row: PgRow = sqlx::query(
        r#"
        SELECT * 
        FROM needs
        WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_one(pool)
    .await
    .unwrap();

    tracing::debug!("Querying DB {}", id);

    // Store the result into Client
    Client {
        row_id: row.get("id"),
        age: row.get("age"),
        gender: row.get("gender"),
        family_members: row.get("family_members"),
        financial_education: row.get("financial_education"),
        income: row.get("income"),
        wealth: row.get("wealth"),
        income_investment: row.get("income_investment"),
        accumulation_investment: row.get("accumulation_investment"),
        client_id: row.get("client_id"),
    }
}

fn preprocessing(client: Client) -> (Product, Vec<f32>) {
    tracing::debug!("Preprocessing client {:?}", client.row_id);

    // Cast the data into floats and store them in a vec
    let client_vec = [
        client.age as f32,
        client.gender as f32,
        client.family_members as f32,
        client.financial_education,
        (client.income.powf(FITTED_LAMBDA_INCOME) - 1.0) / FITTED_LAMBDA_INCOME,
        (client.wealth.powf(FITTED_LAMBDA_WEALTH) - 1.0) / FITTED_LAMBDA_WEALTH,
        client.income_investment as f32,
        client.accumulation_investment as f32,
        client.financial_education * client.wealth.ln(),
    ];

    let product = Product {
        row_id: client.row_id,
        income_investment: client.income_investment,
        accumulation_investment: client.accumulation_investment,
        financial_status: *client_vec.last().unwrap(),
    };

    // Scale the data using means and scales
    let client_vec = client_vec
        .iter()
        .zip(SCALER_MEANS.iter().zip(SCALER_SCALES.iter()))
        .map(|(&value, (&mean, &scale))| (value - mean) / scale)
        .collect();

    tracing::debug!("Preprocessed client {:?}", client.row_id);

    (product, client_vec)
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().unwrap();

    let (config, args) = RuntimeConfig::from_args();
    config.spawn_remote_workers();
    tracing_subscriber::fmt::init();

    let ctx = StreamContext::new(config.clone());
    let limit = args
        .get(1)
        .map(|n| n.parse().expect("expected number for LIMIT"))
        .unwrap_or(5000);

    // let conn_url =
    //     std::env::var("DATABASE_URL").expect("Env var DATABASE_URL is required for this example.");
    let conn_url = "postgresql://postgres:localhostpostgrespass@localhost:5432/postgres";
    let pool = PgPool::connect(&conn_url).await.unwrap();
    let pool_clone = pool.clone();

    // Channel source
    let (tx_channel, source) = ChannelSource::new(32, Replication::One);
    let host_id = config.host_id().unwrap();

    static START: OnceLock<Instant> = OnceLock::new();
    // Host 0 receives the notifications and adds them to the channel
    if host_id == 0 {
        // Start listening channel table_insert
        let mut listener = PgListener::connect(&conn_url).await.unwrap();
        listener.listen("table_insert").await.unwrap();

        tokio::spawn(async move {
            let mut s = listener.into_stream().take(limit);

            // Static atomic counter to debug the number of events coming in
            // including all threads (should be one thread only, but it's best
            // to check)
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            tracing::info!("started postgres listener");
            while let Some(n) = s.next().await {
                let n = n.unwrap();

                // Start measuring from first notification
                START.set(Instant::now()).ok();

                tx_channel.send_async(n).await.unwrap();
                tracing::info!(
                    "recv {}",
                    COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                )
            }
        });
    }

    // Load the model
    let model = Model::<Backend>::default();
    let device = Device::default();

    let count = ctx
        .stream(source)
        .batch_mode(BatchMode::adaptive(97, Duration::from_millis(4000)))
        .map(|notification| {
            let payload = notification.payload();
            let payload: Value = serde_json::from_str(payload).unwrap();

            let id = payload["id"].as_i64().unwrap() as i32;
            let client_id: i32 = payload["client_id"].as_i64().unwrap() as i32;

            tracing::debug!(
                "Received notification with ID: {} and ClientID: {}",
                id,
                client_id
            );

            (id, client_id)
        })
        .group_by(|&(_id, client_id)| client_id)
        .rich_map({
            let mut counter = 0;
            move |(_key, (id, client_id))| {
                counter += 1;
                tracing::debug!("I have client {}, counted {}", client_id, counter);

                (counter, id)
            }
        })
        .filter(|&(_key, (counter, _id))| counter <= MAX_REQUEST)
        .drop_key()
        .map_async(move |(_counter, id)| {
            let p = pool.clone();
            async move {
                let client: Client = get_client(id, &p).await;
                tracing::debug!("Retrieved client {:?}", client.row_id);
                client
            }
        })
        .map(|client| preprocessing(client))
        .map(move |(product, vec)| {
            tracing::debug!("Creating tensor {}", product.row_id);
            let tensor = Tensor::<Backend, 2>::from_data(
                TensorData::new(vec.clone(), [1, vec.clone().len()]),
                &device,
            );
            tracing::debug!("Created tensor {}", product.row_id);
            (product, tensor)
        })
        .map(move |(product, tensor)| {
            tracing::debug!("Predicting {}", product.row_id);
            let prediction = *model
                .forward(tensor)
                .to_data()
                .to_vec::<f32>()
                .unwrap()
                .first()
                .unwrap();
            tracing::debug!("Predicted {}", product.row_id);
            (product, prediction)
        })
        .map_async(move |(product, prediction)| {
            let p: Pool<Postgres> = pool_clone.clone();
            async move {
                let id = product.row_id;
                tracing::debug!("Updating client {}", id);
                update_needs_get_products(prediction, product, &p).await;
                tracing::debug!("Updated client {}", id);
            }
        })
        .collect_count();

    ctx.execute().await;

    if let Some(count) = count.get() {
        println!("{count} passed");
    }

    if let Some(start) = START.get() {
        println!("{:?}", start.elapsed());
    }

    tracing::info!("completed");
}
