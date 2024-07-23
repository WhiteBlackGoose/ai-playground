#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example

use std::time::Duration;

use ab_glyph::{FontRef, PxScale};
use eframe::egui;
use egui::{ColorImage, Image, ImageData, ImageSource, Painter, Rect, TextureHandle};
use image::{imageops::FilterType, GenericImageView, ImageBuffer, Rgb};
use imageproc::{
    drawing::{
        draw_filled_rect_mut, draw_hollow_circle_mut, draw_hollow_rect, draw_hollow_rect_mut,
        draw_text, draw_text_mut,
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

#[derive(Clone, Copy, Debug)]
struct BBox {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl BBox {
    fn to_xyxy(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.x + self.w, self.y + self.h)
    }

    fn intersection(&self, other: BBox) -> f32 {
        let (x1, y1, x2, y2) = self.to_xyxy();
        let (x3, y3, x4, y4) = other.to_xyxy();
        let w = x2.min(x4) - x1.max(x3);
        let h = y2.min(y4) - y1.max(y3);
        if w < 0.0 || h < 0.0 {
            return 0.0;
        }
        w * h
    }

    fn area(&self) -> f32 {
        self.w * self.h
    }

    fn union(&self, other: BBox) -> f32 {
        self.area() + other.area() - self.intersection(other)
    }

    fn iou(&self, other: BBox) -> f32 {
        self.intersection(other) / self.union(other)
    }

    fn shift(&self, x: f32, y: f32) -> Self {
        Self {
            x: self.x + x,
            y: self.y + y,
            w: self.w,
            h: self.h,
        }
    }

    fn scale(&self, scale_x: f32, scale_y: f32) -> Self {
        Self {
            x: self.x * scale_x,
            y: self.y * scale_y,
            w: self.w * scale_x,
            h: self.h * scale_y,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Palm {
    bbox: BBox,
    tips: [(f32, f32); 7],
    score: f32,
}

impl Palm {
    fn shift(&self, x: f32, y: f32) -> Self {
        Self {
            bbox: self.bbox.shift(x, y),
            tips: self.tips.map(|(xt, yt)| (xt + x, yt + y)),
            score: self.score,
        }
    }

    fn scale(&self, scale_x: f32, scale_y: f32) -> Self {
        Self {
            bbox: self.bbox.scale(scale_x, scale_y),
            tips: self.tips.map(|(xt, yt)| (xt * scale_x, yt * scale_y)),
            score: self.score,
        }
    }
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

    fn get_palm(
        &self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        score_threshold: f32,
        iou_threshold: f32,
    ) -> Vec<Palm> {
        let resized = image::imageops::resize(img, 192, 192, FilterType::Triangle);
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
                    offsets[*n] =
                        cell_width as f32 * ((j / repeats / rows) as f32 - (rows - 1) as f32 * 0.5);
                    *n += 3;
                }
            }
        }

        let offsets = get_grid_box_coords();
        let anchors = Array2::from_shape_vec((offsets.len() / 4, 4), offsets).unwrap();

        let regressors = outputs.swap_remove(0).into_shape((1, 2016, 18)).unwrap();
        let scores = outputs.swap_remove(0).into_shape((1, 2016, 1)).unwrap();
        let box_coords = regressors
            .slice(s![0, .., 0..4])
            .into_owned()
            .into_shape((2016, 4))
            .unwrap();
        let box_coords = box_coords + anchors;

        let scale_x = img.width() as f32 / 192.0;
        let scale_y = img.height() as f32 / 192.0;

        let mut palms = (0..2016)
            .map(|i| Palm {
                bbox: BBox {
                    x: box_coords[(i, 0)] - box_coords[(i, 2)] / 2.0,
                    y: box_coords[(i, 1)] - box_coords[(i, 3)] / 2.0,
                    w: box_coords[(i, 2)],
                    h: box_coords[(i, 3)],
                },
                tips: (0..7)
                    .map(|j| {
                        (
                            regressors[(0, i, 4 + j * 2)] + box_coords[(i, 0)],
                            regressors[(0, i, 4 + j * 2 + 1)] + box_coords[(i, 1)],
                        )
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                score: scores[(0, i, 0)],
            })
            .collect::<Vec<_>>();

        palms.sort_by(|p1, p2| p2.score.total_cmp(&p1.score));
        let mut res = vec![];
        loop {
            let palm = palms[0];
            if palm.score < score_threshold {
                break;
            }
            res.push(palm);
            palms.retain(|p2| p2.bbox.iou(palm.bbox) < iou_threshold);
        }

        res.iter()
            .map(|palm| palm.shift(192.0 / 2.0, 192.0 / 2.0).scale(scale_x, scale_y))
            .collect()
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let buf = Self::get_pixel_data(&mut self.camera);

            let palms = self.get_palm(&buf, 0.6, 0.25);
            let mut buf =
                imageproc::image::ImageBuffer::<imageproc::image::Rgb<u8>, Vec<u8>>::from_vec(
                    buf.width(),
                    buf.height(),
                    buf.to_vec(),
                )
                .unwrap();
            for palm in palms {
                draw_hollow_rect_mut(
                    &mut buf,
                    imageproc::rect::Rect::at(palm.bbox.x as i32, palm.bbox.y as i32)
                        .of_size((palm.bbox.w as u32).max(1), (palm.bbox.h as u32).max(1)),
                    imageproc::image::Rgb([255, 255, 0]),
                );
                for (i, (x, y)) in palm.tips.iter().enumerate() {
                    let font =
                        FontRef::try_from_slice(include_bytes!("../DejaVuSans.ttf")).unwrap();
                    let height = 28.0;
                    let scale = PxScale {
                        x: height * 2.0,
                        y: height,
                    };

                    draw_text_mut(
                        &mut buf,
                        imageproc::image::Rgb([255, 0, 255]),
                        *x as i32,
                        *y as i32,
                        scale,
                        &font,
                        &format!("{}", i),
                    );
                }
            }

            let img =
                egui::ColorImage::from_rgb([buf.width() as usize, buf.height() as usize], &buf);
            self.handle.set(img, egui::TextureOptions::LINEAR);
            let txt = egui::load::SizedTexture::from_handle(&self.handle);
            ui.add(egui::Image::from_texture(txt).shrink_to_fit());

            ctx.request_repaint();
        });
    }
}
