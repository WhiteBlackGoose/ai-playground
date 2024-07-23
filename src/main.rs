#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example

use eframe::egui;
use egui::{ColorImage, Image, ImageData, ImageSource, Painter, Rect, TextureHandle};
use image::{imageops::FilterType, GenericImageView, ImageBuffer, Rgb};
use imageproc::{
    drawing::{
        draw_filled_rect_mut, draw_hollow_circle_mut, draw_hollow_rect, draw_hollow_rect_mut,
        draw_text,
    },
    image::Pixel,
};
use ndarray::{s, Array2, Array3, Array4, Axis, Dim};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType},
    Camera,
};
use ort::{ExecutionProvider, SessionBuilder, Tensor};

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
    fn get_pixel_data(camera: &mut Camera) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let frame = camera.frame().unwrap();
        let buf = frame.decode_image::<RgbFormat>().unwrap();
        buf
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
        println!("{:?}", session.outputs);

        let index = CameraIndex::Index(0);
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut camera = Camera::new(index, requested).unwrap();
        camera.open_stream().unwrap();

        let buf = Self::get_pixel_data(&mut camera);
        let img = egui::ColorImage::from_rgb(
            [buf.width() as usize, buf.height() as usize],
            &buf.to_vec(),
        );
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
            let buf = Self::get_pixel_data(&mut self.camera);

            let resized = image::imageops::resize(&buf, 192, 192, FilterType::Triangle);
            let imgnd = Array4::from_shape_vec(
                (1, 192, 192, 3),
                resized
                    .iter()
                    .map(|v| *v as f32 / 255.0)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let tensor = Tensor::from_array(imgnd).unwrap();
            let outputs = self.session.run(ort::inputs![tensor].unwrap()).unwrap();
            let mut outputs = outputs
                .iter()
                .map(|o| o.1.try_extract_tensor::<f32>().unwrap().view().into_owned())
                .collect::<Vec<_>>();

            assert_eq!(outputs.len(), 2);

            fn get_grid_box_coords() -> Vec<f32> {
                let mut offsets = vec![0.0; 2016 * 4];
                let mut n = 0usize;
                add_grid(&mut offsets, 24, 2, 8, &mut n);
                add_grid(&mut offsets, 12, 6, 16, &mut n);
                return offsets;

                fn add_grid(
                    offsets: &mut [f32],
                    rows: usize,
                    repeats: usize,
                    cell_width: usize,
                    n: &mut usize,
                ) {
                    for j in 0..repeats * rows * rows {
                        offsets[*n] = cell_width as f32
                            * (((j / repeats) % rows) as f32 - (rows - 1) as f32 * 0.5);
                        *n += 1;
                        offsets[*n] = cell_width as f32
                            * ((j / repeats / rows) as f32 - (rows - 1) as f32 * 0.5);
                        *n += 3;
                    }
                }
            }

            let offsets = get_grid_box_coords();
            let anchors = Array2::from_shape_vec((offsets.len() / 4, 4), offsets).unwrap();

            let regressors = outputs.swap_remove(0).into_shape((1, 2016, 18)).unwrap();
            let scores = outputs.swap_remove(0);
            let box_coords = regressors.slice(s![0, .., 0..4]).into_owned() + anchors;

            let scale_x = buf.width() as f32 / 192.0;
            let scale_y = buf.height() as f32 / 192.0;

            let mut scores = scores.iter().enumerate().collect::<Vec<_>>();
            scores.sort_by(|x, y| y.1.total_cmp(x.1));
            let mut buf =
                imageproc::image::ImageBuffer::<imageproc::image::Rgb<u8>, Vec<u8>>::from_vec(
                    buf.width(),
                    buf.height(),
                    buf.to_vec(),
                )
                .unwrap();
            for (index, score) in &scores[..4] {
                let index = *index;
                let score = **score;
                if score < 0.8 {
                    break;
                }
                let (x, y, w, h) = (
                    box_coords[(index, 0)] + 192.0 / 2.0,
                    box_coords[(index, 1)] + 192.0 / 2.0,
                    box_coords[(index, 2)],
                    box_coords[(index, 3)],
                );
                let (x, y, w, h) = (x * scale_x, y * scale_y, w * scale_x, h * scale_y);
                let (x, y, w, h) = (
                    (x - w / 2.0) as i32,
                    (y - h / 2.0) as i32,
                    w as u32,
                    h as u32,
                );

                draw_hollow_rect_mut(
                    &mut buf,
                    imageproc::rect::Rect::at(x, y).of_size(w.max(1), h.max(1)),
                    imageproc::image::Rgb([255, 255, 0]),
                );
                for i in 0..7 {
                    let x = x
                        + ((regressors[(0, index, 4 + i * 2)] - regressors[(0, index, 0)])
                            * scale_x) as i32;
                    let y = y
                        + ((regressors[(0, index, 4 + i * 2 + 1)] - regressors[(0, index, 1)])
                            * scale_x) as i32;
                    for t in 13..15 {
                        draw_hollow_circle_mut(
                            &mut buf,
                            (x, y),
                            t,
                            imageproc::image::Rgb([255, 0, 255]),
                        );
                    }
                }
            }

            let img =
                egui::ColorImage::from_rgb([buf.width() as usize, buf.height() as usize], &buf);
            self.handle.set(img, egui::TextureOptions::LINEAR);
            let txt = egui::load::SizedTexture::from_handle(&self.handle);
            ui.add(egui::Image::from_texture(txt).shrink_to_fit());

            for (_, score) in &scores[..4] {
                ui.label(format!("Score: {:.2}", *score));
            }
            ctx.request_repaint();
        });
    }
}
