use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/deep_model.onnx") // Path to your ONNX model
        .out_dir("model/")               // Directory where the generated Rust code will be placed
        .run_from_script();              // Execute the model generation
}
