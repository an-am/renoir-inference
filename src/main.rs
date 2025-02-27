use std::sync::mpsc;
use regex::Regex;
use renoir::prelude::*;
mod model;
use burn::tensor::*;
use sqlx::{postgres::{PgListener, PgNotification}, FromRow};
use burn_ndarray::{NdArray, NdArrayDevice};
use model::deep_model::Model;
use serde::{Serialize, Deserialize};
use postgres::{Client as PgClient, NoTls};
use tokio::spawn;

const MAX_REQUEST: i32 = 5;
const NUM_CLIENTS: i32 = 100;
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

fn update_needs_get_products(prediction: f32, product: Product, database_url: &str) -> () {
    // Connect to the DB
    println!("Model output for id {}: {}", product.row_id, prediction);
    
    let mut conn = PgClient::connect(database_url, NoTls).unwrap();
    conn.execute(
        "UPDATE needs
                SET risk_propensity = $1, financial_status = $2
                WHERE id = $3", &[&prediction, &product.financial_status, &product.row_id]).unwrap();
    // Retrieve products from product table
    let products = conn.query(
        "SELECT id_product, description 
                FROM products
                WHERE   
                    (
                        income  = $1
                        OR accumulation = $2
                    )
                    AND risk <= $3", &[&product.income_investment, 
                &product.accumulation_investment, &prediction]).unwrap();
    
    println!("Advised {} products for id={}", products.len(), product.row_id);

    for p in products {
        let id_product: i32 = p.get("id_product");
        let description: &str = p.get("description");

        println!("ID: {}, description: {}", id_product, description);
    }
    
}

fn get_client(id: i32, database_url: &str) -> Client {
    // Connect to the DB
    let mut conn = PgClient::connect(database_url, NoTls).unwrap();

    // Retrieve the row through its id
    let row = conn.query_one(
        "SELECT id, age, gender, family_members, financial_education, 
                    income, wealth, income_investment, accumulation_investment, client_id
                FROM needs WHERE id = $1", 
                &[&id]
            ).unwrap();
    
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
        client_id: row.get("client_id") 
    }
}

fn preprocessing(client: Client) -> (Product, Vec<f32>) {

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
    (product, client_vec)
}

#[tokio::main]
async fn main() {
    let (config, _args) = RuntimeConfig::from_args();
    config.spawn_remote_workers();
    let ctx = StreamContext::new(config);

    let database_url = "postgres://postgres@localhost:5432/postgres";

    // Start listening channel table_insert
    let mut listener= PgListener::connect(&database_url).await.unwrap();
    let _ = listener.listen("table_insert").await;  

    // Load the model
    let model = Model::<NdArray<f32>>::default();

    // Create a synchronous channel.
    let (sender, receiver) = mpsc::channel::<PgNotification>();
    
    // Spawn an async task to continuously receive notifications.
    spawn(async move {
        loop {
            match listener.recv().await {
                Ok(notification) => {
                    // Send the notification over the channel.
                    if sender.send(notification).is_err() {
                        eprintln!("Receiver dropped. Exiting notification loop.");
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Error receiving notification: {:?}", e);
                    break;
                }
            }
        }
    });
    
    let notifications_iter = receiver.into_iter();

    ctx.stream_iter(notifications_iter)
        .map(|notification|{
            // Get the notification payload
            let payload = notification.payload();

            // Get id and client_id from the payload
            let re = Regex::new(r#"\{"id"\s*:\s*(\d+),\s*"client_id"\s*:\s*(\d+)\}"#).unwrap();
            let caps = re.captures(&payload).unwrap();
            let id: i32 = caps.get(1).unwrap().as_str().parse().unwrap();
            let client_id: i32 = caps.get(2).unwrap().as_str().parse().unwrap();

            println!("Received notification with ID: {} and ClientID: {}", id, client_id);

            (id, client_id)
        })
        .group_by(|&(_id, client_id)| client_id.clone() % NUM_CLIENTS)
        .rich_map({
            let mut counter = 0;
            move |(_key, (id ,_client_id))| {
                counter += 1;
                (counter, id)
        }})
        .filter(|&(_key, (counter, _id))| counter <= MAX_REQUEST)
        .drop_key()
        .map(|(_counter, id)| get_client(id, database_url))
        .map(|client| preprocessing(client))
        .map(|(product, vec)| {
            let tensor = Tensor::<Backend, 2>::from_data(
                TensorData::new(vec.clone(), [1, vec.clone().len()]), 
                &NdArrayDevice::default());
            (product, tensor)
            }
        )
        .map(move |(product, tensor)| {  
            //let model = Model::<NdArray<f32>>::default();
            let prediction = model.forward(tensor);
            (product, prediction)
        })
        .map(|(product, prediction)| (product, *prediction.to_data().to_vec::<f32>().unwrap().first().unwrap()))
        .for_each(|(product, prediction)| update_needs_get_products(prediction, product, database_url));  

    ctx.execute().await;
}