#!/bin/bash/

echo "Run ConvoSentinel on $1"

device=1
echo "Using device $device. Change the device number in the script if you want to use a different device."

echo "Run message-level SI detector and save results to $2"
cd ./message_si_detector
python run_si_detector.py --device $device \
    --model_checkpoint ./trained_si_model \
    --input_file ../$1 --output_file ../$2
cd ..

echo "Run message-level SE attempt detector and save results to $3"
cd ./se_attempt_detector
python llama_msg_level.py --device $device --test_file ../$1 --si_file ../$2 --prediction_file ../$3

echo "Run conversation-level SE attempt detector and save results to $4"
python gpt_conv_level.py --test_file ../$1 --si_file ../$2 --msg_label_file ../$3 --prediction_file ../$4

echo "Conversation-level SE attempt detection results:"
python evaluate_conv_predictions.py --prediction_file ../$4
cd ..

echo "Finished!"
