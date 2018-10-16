//
//  ViewController.swift
//  Tiramisu
//
//  Created by James Kauten on 10/15/18.
//  Copyright © 2018 Kautenja. All rights reserved.
//

import UIKit
import AVFoundation
import Vision

/// A view controller to pass camera inputs through a vision model
class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    /// the view to preview raw RGB data from the camera
    @IBOutlet weak var preview: UIView!
    /// the view for showing the segmentation
    @IBOutlet weak var segmentation: UIImageView!

    /// the camera session for streaming data from the camera
    var captureSession: AVCaptureSession!
    /// the output from the video
    var videoOutput: AVCaptureVideoDataOutput!
    /// the video preview layer
    var videoPreviewLayer: AVCaptureVideoPreviewLayer!

    /// the model for the view controller to apss camera data through
    var model: VNCoreMLModel!

    /// Handle a callback from the view loading
    override func viewDidLoad() {
        super.viewDidLoad()
        if model == nil {
            // TODO: handle with a try fail block
            model = try! VNCoreMLModel(for: Tiramisu45().model)
        }
    }

    /// Handle the view appearing
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // setup the AV session
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .medium
        // get a handle on the back camera
        guard let camera = AVCaptureDevice.default(for: AVMediaType.video)
            else {
                // TODO: handle better
                print("Unable to access back camera!")
                return
        }
        // create an input device from the back camera and handle
        // any errors (i.e., privacy request denied)
        do {
            // setup the camera input and video output
            let input = try AVCaptureDeviceInput(device: camera)
            videoOutput = AVCaptureVideoDataOutput()
            videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            // add the inputs and ouptuts to the sessionr and start the preview
            if captureSession.canAddInput(input) && captureSession.canAddOutput(videoOutput) {
                captureSession.addInput(input)
                captureSession.addOutput(videoOutput)
                setupLivePreview()
            }
        }
        catch let error  {
            // TODO: handle better
            print("Error Unable to initialize back camera:  \(error.localizedDescription)")
            return
        }
    }

    /// Setup the Live preview from the camera
    func setupLivePreview() {
        // create a video preview layer for the view controller
        videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        // set the metadata of the video preview
        videoPreviewLayer.videoGravity = .resizeAspect
        videoPreviewLayer.connection?.videoOrientation = .landscapeRight
        // add the preview layer as a sublayer of the preview view
        preview.layer.addSublayer(videoPreviewLayer)
        // start the capture session asyncrhonously
        DispatchQueue.global(qos: .userInitiated).async {
            // start the capture session in the background thread
            self.captureSession.startRunning()
            // set the frame of the video preview to the bounds of the
            // preview view
            DispatchQueue.main.async {
                self.videoPreviewLayer.frame = self.preview.bounds
            }
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // run an inference with CoreML
        let request = VNCoreMLRequest(model: model) { (finishedRequest, error) in
            // handle an error from the inference engine
            if let _ = error {
                print("inference error")
                return
            }
            // get the inference results
            guard let outputs = finishedRequest.results as? [VNCoreMLFeatureValueObservation] else {
                print("failed to get inference results")
                return
            }
            // get the multi array from the outputs of the model
            guard let multiArray = outputs[0].featureValue.multiArrayValue else {
                print("failed to cast inference to expected type")
                return
            }
            // unmap the discrete segmentation to RGB pixels
            let image = probsToImage(multiArray)
            DispatchQueue.main.async {
                self.segmentation.image = image
            }
        }
        // create a Core Video pixel buffer which is an image buffer that holds pixels in main memory
        // Applications generating frames, compressing or decompressing video, or using Core Image
        // can all make use of Core Video pixel buffers
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        // execute the request
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    }
    
}

/// Convert probability tensor into an image
func probsToImage(_ _probs: MLMultiArray) -> UIImage? {
    // TODO: dynamically load a label map instead of hard coding
    // can this bonus data be included in the model file?
    let label_map = [
        0:  [0, 128, 192],
        1:  [128, 0, 0],
        2:  [64, 0, 128],
        3:  [192, 192, 128],
        4:  [64, 64, 128],
        5:  [64, 64, 0],
        6:  [128, 64, 128],
        7:  [0, 0, 192],
        8:  [192, 128, 128],
        9:  [128, 128, 128],
        10: [192, 192, 0],
        11: [0, 0, 0]
    ]
    // convert the MLMultiArray to a MultiArray
    var probs = MultiArray<Double>(_probs)
    // get the shape information from the probs
    let classes = probs.shape[0]
    let height = probs.shape[1]
    let width = probs.shape[2]
    // initialize some bytes to store the image in
    var bytes = [UInt8](repeating: 0, count: height * width * 4)
    // iterate over the pixels in the output probs
    for h in 0 ..< height {
        for w in 0 ..< width {
            // store the highest probability and the corresponding class
            var max_prob: Double = 0
            var max_c: Int = 0
            // iterate over class labels to extract the highest probability
            for c in 0 ..< classes {
                // replace the highest prob and index if this prob is greater
                if probs[c, h, w] > max_prob {
                    max_prob = probs[c, h, w]
                    max_c = c
                }
            }
            // get the array offset for this word
            let offset = h * width * 4 + w * 4
            // get the RGB value for the highest probability class
            let rgb = label_map[max_c]
            // set the bytes to the RGB value and alpha of 1.0 (255)
            bytes[offset + 0] = UInt8(rgb![0])
            bytes[offset + 1] = UInt8(rgb![1])
            bytes[offset + 2] = UInt8(rgb![2])
            bytes[offset + 3] = 255
        }
    }
    // create a UIImage from the byte array
    return UIImage.fromByteArray(bytes, width: width, height: height,
                                 scale: 0, orientation: .up,
                                 bytesPerRow: width * 4,
                                 colorSpace: CGColorSpaceCreateDeviceRGB(),
                                 alphaInfo: .premultipliedLast)
}
