# https://forums.developer.apple.com/thread/84401
import coremltools
import sys

def update_multiarray_to_float32(feature):
    if feature.type.HasField('multiArrayType'):
        import coremltools.proto.FeatureTypes_pb2 as _ft
        feature.type.multiArrayType.dataType = _ft.ArrayFeatureType.FLOAT32

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: %s <input_model_path> <output_model_path>" % sys.argv[0])
        sys.exit(1)

    input_model_path = sys.argv[1]
    output_model_path = sys.argv[2]

    spec = coremltools.utils.load_spec(input_model_path)

    for input_feature in spec.description.input:
        update_multiarray_to_float32(input_feature)

    for output_feature in spec.description.output:
        update_multiarray_to_float32(output_feature)

    coremltools.utils.save_spec(spec, output_model_path)
