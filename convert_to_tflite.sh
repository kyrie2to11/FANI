nohup python convert.py \
--model_path model/fani.py \
--model_name FANI \
--input_shapes 1,180,320,9 \
--ckpt_path snapshot/ckpt-24 \
--output_tflite tflite/fani.tflite &
