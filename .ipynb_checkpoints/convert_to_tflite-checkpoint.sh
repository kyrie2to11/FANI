nohup python convert.py \
--model_path model/vsr.py \
--model_name MVSR \
--input_shapes 1,180,320,9 \
--ckpt_path snapshot/ckpt-1 \
--output_tflite tflite/mvsr_7.tflite &
