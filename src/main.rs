#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example

use eframe::egui;
use egui::{ColorImage, Image, ImageData, ImageSource, TextureHandle};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType},
    Camera,
};
use ort::{ExecutionProvider, SessionBuilder};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 960.0]),
        ..Default::default()
    };
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|cc| {
            // This gives us image support:
            egui_extras::install_image_loaders(&cc.egui_ctx);

            Ok(Box::new(MyApp::new(&cc.egui_ctx)))
        }),
    )
}

struct MyApp {
    camera: Camera,
    session: ort::Session,
    handle: TextureHandle,
}

impl MyApp {
    fn get_pixel_data(camera: &mut Camera) -> (Vec<u8>, usize, usize) {
        let frame = camera.frame().unwrap();
        let buf = frame.decode_image::<RgbFormat>().unwrap();
        let v = buf.to_vec();
        (v, buf.width() as usize, buf.height() as usize)
    }

    fn new(ctx: &egui::Context) -> Self {
        let builder = SessionBuilder::new().unwrap();
        let cuda = ort::CUDAExecutionProvider::default();
        match cuda.register(&builder) {
            Ok(_) => println!("CUDA found"),
            Err(e) => println!("Cuda not found!\n{}", e),
        }
        let session = builder
            .commit_from_file("./palm_detection_lite.onnx")
            .unwrap();

        // first camera in system
        let index = CameraIndex::Index(0);
        // request the absolute highest resolution CameraFormat that can be decoded to RGB.
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        // make the camera
        let mut camera = Camera::new(index, requested).unwrap();
        camera.open_stream().unwrap();

        let (buf, width, height) = Self::get_pixel_data(&mut camera);
        let img = egui::ColorImage::from_rgb([width, height], &buf);
        Self {
            camera,
            session,
            handle: ctx.load_texture("s", img, egui::TextureOptions::LINEAR),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (buf, width, height) = Self::get_pixel_data(&mut self.camera);
            let img = egui::ColorImage::from_rgb([width, height], &buf);
            self.handle.set(img, egui::TextureOptions::LINEAR);
            let txt = egui::load::SizedTexture::from_handle(&self.handle);
            ui.add(egui::Image::from_texture(txt).shrink_to_fit());

            // ui.add(Image::from_bytes("", img));
        });
    }
}
