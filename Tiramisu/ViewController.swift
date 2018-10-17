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
import Metal
import MetalPerformanceShaders

/// A view controller to pass camera inputs through a vision model
class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    /// a local reference to time to update the framerate
    var time = Date()

    /// the view to preview raw RGB data from the camera
    @IBOutlet weak var preview: UIView!
    /// the view for showing the segmentation
    @IBOutlet weak var segmentation: UIImageView!
    /// a label to show the framerate of the model
    @IBOutlet weak var framerate: UILabel!
    
    /// the camera session for streaming data from the camera
    var captureSession: AVCaptureSession!
    /// the video preview layer
    var videoPreviewLayer: AVCaptureVideoPreviewLayer!
    
    /// TODO:
    private var _device: MTLDevice?
    /// TODO:
    var device: MTLDevice! {
        get {
            // try to unwrap the private model instance
            if let device = _device {
                return device
            }
            // try to create a new model and fail gracefully
            _device = MTLCreateSystemDefaultDevice()
            return _device
        }
    }

    /// the model for the view controller to apss camera data through
    private var _model: VNCoreMLModel?
    /// the model for the view controller to apss camera data through
    var model: VNCoreMLModel! {
        get {
            // try to unwrap the private model instance
            if let model = _model {
                return model
            }
            // try to create a new model and fail gracefully
            do {
                _model = try VNCoreMLModel(for: Tiramisu45().model)
            } catch let error {
                let message = "failed to load model: \(error.localizedDescription)"
                popup_alert(self, title: "Model Error", message: message)
            }
            return _model
        }
    }
    
    /// the request and handler for the model
    private var _request: VNCoreMLRequest?
    /// the request and handler for the model
    var request: VNCoreMLRequest! {
        get {
            // try to unwrap the private request instance
            if let request = _request {
                return request
            }
            // create the request
            _request = VNCoreMLRequest(model: model) { (finishedRequest, error) in
                // handle an error from the inference engine
                if let error = error {
                    print("inference error: \(error.localizedDescription)")
                    return
                }
                // get the outputs from the model
                let outputs = finishedRequest.results as? [VNCoreMLFeatureValueObservation]
                // get the probabilities as the first output of the model
                guard let probs = outputs?[0].featureValue.multiArrayValue else {
                    print("failed to extract output from model")
                    return
                }
                let channels = probs.shape[0].intValue
                let height = probs.shape[1].intValue
                let width = probs.shape[2].intValue
                                
                let filter = MPSNNReduceFeatureChannelsArgumentMax(device: self.device)
                let desc = MPSImageDescriptor(channelFormat: .float32,
                                              width: width,
                                              height: height,
                                              featureChannels: channels)
                let mps_image = MPSImage(device: self.device, imageDescriptor: desc)
                mps_image.writeBytes(probs.dataPointer,
                                     dataLayout: .featureChannelsxHeightxWidth,
                                     imageIndex: 0)
                
//                var featureChannelInfo = MPSImageReadWriteParams()
//                featureChannelInfo.numberOfFeatureChannelsToReadWrite = 12
//                let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: width, height: height, depth: 1))
//                mps_image.readBytes(probs.dataPointer,
//                                    dataLayout: .featureChannelsxHeightxWidth,
//                                    bytesPerRow: width * channels * MemoryLayout<Double>.size,
//                                    region: region,
//                                    featureChannelInfo: featureChannelInfo,
//                                    imageIndex: 0)
                

//                let buffer = self.device.makeCommandQueue()?.makeCommandBuffer()
//                let classes = filter.encode(commandBuffer: buffer!, sourceImage: mps_image)
//                let desc1 = MPSImageDescriptor(channelFormat: .unorm8, width: width, height: height, featureChannels: 1)
//                let classes = MPSImage(device: self.device, imageDescriptor: desc1)
//                filter.encode(commandBuffer: buffer!, sourceImage: mps_image, destinationImage: classes)
//                print(classes.width)
//                print(classes.height)
//                print(classes.featureChannels)
//                print()
//                // set the read count to zero to release the memory
//                if let _classes = classes as? MPSTemporaryImage { _classes.readCount = 0 }
                let argmax = try! MLMultiArray(shape: [12, probs.shape[1], probs.shape[2]], dataType: .float32)
                mps_image.readBytes(argmax.dataPointer,
                                    dataLayout: .featureChannelsxHeightxWidth,
                                    imageIndex: 0)

                // unmap the discrete segmentation to RGB pixels
                let image = probsToImage(argmax)
                // update the image on the UI thread
                DispatchQueue.main.async {
                    self.segmentation.image = image
                    let fps = -1 / self.time.timeIntervalSinceNow
                    self.time = Date()
                    self.framerate.text = "\(fps)"
                }
            }
            // set the input image size to be a scaled version
            // of the image
            _request?.imageCropAndScaleOption = .scaleFill
            return _request
        }
    }
    
    /// Respond to a memory warning from the OS
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        popup_alert(self, title: "Memory Warning", message: "received memory warning")
    }
    
    /// Handle the view appearing
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // setup the AV session
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .hd1280x720
        // get a handle on the back camera
        guard let camera = AVCaptureDevice.default(for: AVMediaType.video) else {
            let message = "Unable to access the back camera!"
            popup_alert(self, title: "Camera Error", message: message)
            return
        }
        // create an input device from the back camera and handle
        // any errors (i.e., privacy request denied)
        do {
            // setup the camera input and video output
            let input = try AVCaptureDeviceInput(device: camera)
            let videoOutput = AVCaptureVideoDataOutput()
            videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            // add the inputs and ouptuts to the sessionr and start the preview
            if captureSession.canAddInput(input) && captureSession.canAddOutput(videoOutput) {
                captureSession.addInput(input)
                captureSession.addOutput(videoOutput)
                setupCameraPreview()
            }
        }
        catch let error  {
            let message = "failed to intialize camera: \(error.localizedDescription)"
            popup_alert(self, title: "Camera Error", message: message)
            return
        }
    }

    /// Setup the live preview from the camera
    func setupCameraPreview() {
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
    
    /// Handle a frame from the camera video stream
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // create a Core Video pixel buffer which is an image buffer that holds pixels in main memory
        // Applications generating frames, compressing or decompressing video, or using Core Image
        // can all make use of Core Video pixel buffers
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            let message = "failed to create pixel buffer from video input"
            popup_alert(self, title: "Inference Error", message: message)
            return
        }
        // execute the request
        do {
            try VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
        } catch let error {
            let message = "failed to perform inference: \(error.localizedDescription)"
            popup_alert(self, title: "Inference Error", message: message)
        }
    }
    
}
