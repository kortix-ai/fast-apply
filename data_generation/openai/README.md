# Example

```sh
python batch_processor.py -i data/train2/batch2/ -o data/train2/train_batch_cal_com.parquet
python send_batch_request.py -bd data/train2/batch2/ -c 5
python prepare_batch_data.py -i data/train2/train_cal_com.parquet -o data/train2/batch2/
```