use renoir::prelude::*;
mod model;
use burn::tensor::*;
use burn_ndarray::{NdArray, NdArrayDevice};
use model::deep_model::Model;
use serde::{Serialize, Deserialize};

const FITTED_LAMBDA_INCOME: f32 = 0.3026418664067109;
const FITTED_LAMBDA_WEALTH: f32 =  0.1336735055366279;
const SCALER_MEANS: [f32; 9] = [55.2534, 0.492, 2.5106, 0.41912250425, 7.664003606794818, 5.650795740523163, 0.3836, 0.5132, 1.7884450179129336];
const SCALER_SCALES: [f32; 9] = [11.970496582849016, 0.49993599590347565, 0.7617661320904205, 0.15136821202756753, 2.4818937424620633, 1.5813522545815777, 0.4862623160393987, 0.49982572962983807, 0.8569630982206199];

type Backend = NdArray<f32>;

#[derive(Clone, Deserialize, Serialize, Debug)]
struct Client {
    row_id: i32,
    age: i8,
    gender: i8,
    family_members: i8,
    financial_education: f32,
    risk_propensity: f32,
    income: f32,
    wealth: f32,
    income_investment: i8,
    accumulation_investment: i8,
    client_id: i32,
}


fn preprocessing(client: Client) -> Vec<f32> {
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
    
    // Scale the data using means and scales
    client_vec
        .iter()
        .zip(SCALER_MEANS.iter().zip(SCALER_SCALES.iter()))
        .map(|(&value, (&mean, &scale))| (value - mean) / scale)
        .collect()
}

fn main() {
    // Create a device for tensor computation
    let device = NdArrayDevice::default();

    // Load the model
    let model = Model::<NdArray<f32>>::default();

    let (config, args) = RuntimeConfig::from_args();
    config.spawn_remote_workers();
    let env = StreamContext::new(config);

    let source = CsvSource::<Client>::new("/Users/antonelloamore/VS code/renoir-prediction/Needs.csv");

    let s = env
        .stream(source)
        .map(|v| preprocessing(v))
        .map(move |v| Tensor::<Backend, 2>::from_data(TensorData::new(v.clone(), [1, v.len()]), &device))
        .map(move |v| model.forward(v))
        .map(|v| v.to_data().to_vec::<f32>().unwrap())
        .for_each(|v| println!("Model output: {:?}", v.first().unwrap()));
    
    env.execute_blocking();
}

