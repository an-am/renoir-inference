use renoir::prelude::*;
mod model;
use burn::tensor::*;
use sqlx::{postgres::{PgListener, PgPool, PgRow}, FromRow, Pool, Postgres, Row};
use burn_ndarray::{NdArray, NdArrayDevice};
use renoir::operator::source::ChannelSource;
use model::deep_model::Model;
use serde::{Serialize, Deserialize};
use serde_json::Value;

use std::{fs::File, time::Duration};
use std::io::{self, Write};

const MAX_REQUEST: i32 = 5;
const FITTED_LAMBDA_INCOME: f32 = 0.3026418664067109;
const FITTED_LAMBDA_WEALTH: f32 =  0.1336735055366279;
const SCALER_MEANS: [f32; 9] = [55.2534, 0.492, 2.5106, 0.41912250425, 7.664003606794818, 5.650795740523163, 0.3836, 0.5132, 1.7884450179129336];
const SCALER_SCALES: [f32; 9] = [11.970496582849016, 0.49993599590347565, 0.7617661320904205, 0.15136821202756753, 2.4818937424620633, 1.5813522545815777, 0.4862623160393987, 0.49982572962983807, 0.8569630982206199];

type Backend = NdArray<f32>;

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
    financial_status: f32
}

async fn update_needs_get_products(prediction: f32, product: Product, pool: &PgPool) -> (i32, f32) {
    // Connect to the DB
    // println!("Model output for id {}: {}", product.row_id, prediction);

    //  &[&prediction, &product.financial_status, &product.row_id]
    //let mut conn = PgClient::connect(database_url, NoTls).unwrap();
    let _ = sqlx::query(r#"
        UPDATE needs
        SET risk_propensity = $1, financial_status = $2
        WHERE id = $3
        "#)
        .bind(prediction)
        .bind(product.financial_status)
        .bind(product.row_id)
        .execute(pool)
        .await;

    println!("Querying UPDATE {}", product.row_id);
   
    // Retrieve products from product table
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
        "#
        )
        .bind(product.income_investment)
        .bind(product.accumulation_investment)
        .bind(prediction)
        .fetch_all(pool)
        .await
        .unwrap();
    
    println!("Advised {} products for id={}", products.len(), product.row_id);

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
        "#
        )
        .bind(id)
        .fetch_one(pool)
        .await
        .unwrap();

    println!("Querying DB {}", id);
    
    // Store the result into Client
    let client = Client { 
        row_id: row.get("id"), 
        age: row.get("age"), 
        gender: row.get("gender"), 
        family_members: row.get("family_members"), 
        financial_education: row.get("financial_education"), 
        income: row.get("income"), 
        wealth: row.get("wealth"), 
        income_investment: row.get("income_investment"), 
        accumulation_investment: row.get("accumulation_investment"), 
        client_id: row.get("client_id") 
    };
    client
}

fn preprocessing(client: Client) -> (Product, Vec<f32>) {

    println!("Preprocessing client {:?}", client.row_id);

    // Cast the data into floats and store them in a vec
    let client_vec = vec![
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
    
    println!("Preprocessed client {:?}", client.row_id);
    
    (product, client_vec)
}


#[tokio::main]
async fn main() {
    let (config, _args) = RuntimeConfig::from_args();
    let config_clone = config.clone();
    config.spawn_remote_workers();
    let ctx = StreamContext::new(config);

    let db_url = "postgres://postgres@localhost:5432/postgres";
    std::env::set_var("DATABASE_URL", db_url);

    let conn_url =
        std::env::var("DATABASE_URL").expect("Env var DATABASE_URL is required for this example.");
    let pool = PgPool::connect(&conn_url).await.unwrap();

    //let conn = PgPool::connect(&database_url).await.unwrap();
    let pool_clone = pool.clone();

    // Start listening channel table_insert
    let mut listener= PgListener::connect(&conn_url).await.unwrap();
    let _ = listener.listen("table_insert").await;  

    // Load the model
    let model = Model::<NdArray<f32>>::default();

    // Channel source
    let (tx_channel, source) = ChannelSource::new(10_000 as usize, Replication::One);

    // Host 0 receives the notifications and adds them to the channel
    if config_clone.host_id().unwrap() == 0 {
        tokio::spawn(async move {

            loop {
                let n = listener.recv().await.unwrap();
                let _ = tx_channel.send(n);
                
               /*  match listener.recv().await {
                    Ok(notification) => {
                        // Send the notification to the Renoir source; back‑pressure handled by flume
                        let _ = tx_channel.send(notification);
                    }
                    Err(e) => {
                        eprintln!("Error receiving notification: {e:?}");
                        break;
                    }
                } */
            }
        });
    }

    ctx.stream(source)
        .batch_mode(BatchMode::adaptive(10, Duration::from_millis(100)))
        .map(|notification| {
            // Get the notification payload
            let payload = notification.payload();

            // Get id and client_id from the payload
            let payload: Value = serde_json::from_str(payload).unwrap();
            let id = payload["id"].as_i64().unwrap() as i32;
            let client_id: i32 = payload["client_id"].as_i64().unwrap() as i32;

            println!("Received notification with ID: {} and ClientID: {}", id, client_id);

            (id, client_id)
        })
        .group_by(|&(_id, client_id)| client_id)
        .rich_map({
            let mut counter = 0;
            move |(_key, (id , client_id))| {
                counter += 1;
                println!("I have client {}, counted {}", client_id, counter);

                (counter, id)
        }})
        .filter(|&(_key, (counter, _id))| counter <= MAX_REQUEST)
        .drop_key()
        .map_async(move |(_counter, id)| {
            let p = pool.clone();
            async move { 
                let client: Client = get_client(id, &p).await;
                println!("Retrieved client {:?}", client.row_id);
                client
            }
        })
        .map(|client| preprocessing(client))
        .map(|(product, vec)| {
            println!("Creating tensor {}", product.row_id);
            let tensor = Tensor::<Backend, 2>::from_data(
                TensorData::new(vec.clone(), [1, vec.clone().len()]), 
                &NdArrayDevice::default());
            println!("Created tensor {}", product.row_id);
            (product, tensor)
        })
        .map(move |(product, tensor)| {  
            println!("Predicting {}", product.row_id);
            let prediction = *model
                .forward(tensor)
                .to_data()
                .to_vec::<f32>()
                .unwrap()
                .first()
                .unwrap();
            println!("Predicted {}", product.row_id);
            (product, prediction)
        })
        .map_async(move |(product, prediction)| {
            let p: Pool<Postgres> = pool_clone.clone();
            async move { 
                let id = product.row_id.clone();
                println!("Updating client {}", id);
                update_needs_get_products(prediction, product, &p).await;
                println!("Updated client {}", id);
            }
        })
        .collect_count();
    
    ctx.execute().await;

}