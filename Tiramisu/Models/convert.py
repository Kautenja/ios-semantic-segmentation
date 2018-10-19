"""A script to convert a Keras vision model to a CoreML model."""
import sys
import os
import coremltools


# try to unwrap the path to the weights
try:
    weights = sys.argv[1]
    # create an output file using the same name as input with new extension
    output_file = weights.replace('.h5', '.mlmodel')
except IndexError:
    print(__doc__)


# load the CoreML model from the Keras model
coreml_model = coremltools.converters.keras.convert(weights,
	input_names='image',
	image_input_names='image',
    output_names='segmentation',
	image_scale=1/255.0,
)


# setup the attribution meta-data for the model
coreml_model.author = 'Kautenja'
coreml_model.license = 'MIT'
coreml_model.short_description = '45 Layers Tiramisu Semantic Segmentation Model trained on CamVid & CityScapes.'
coreml_model.input_description['image'] = 'An input image in RGB order'
coreml_model.output_description['segmentation'] = 'The segmentation map as the Softmax output'


# get the spec from the model
spec = coreml_model.get_spec()
# create a local reference to the Float32 type
Float32 = coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
# set the output shape for the segmentation to Float32
spec.description.output[0].type.multiArrayType.dataType = Float32
# save the spec to disk
coremltools.utils.save_spec(spec, output_file)
