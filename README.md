# Customer Booking Prediction with MLOps
<img src="artifacts/britishaiways.jpg" alt="workflow" width="100%">


## About This Project
This project utilizes DVC, an MLOPs tool that assists in Machine Learning pipeline management, and AWS to store project data and artifacts for British Airways customer booking details.

### To run this project, you must have an AWS account.

## Usage
1. Clone this repository to your local machine:
```
git clone https://github.com/titoausten/british-airways-customer-booking-prediction-with-mlops
.git
```

2. Install requirements:
```
pip install -r requirements.txt

```

3. Initialize DVC:
```
dvc init
```

4. Create S3 Bucket for backend storage:
```
python s3_bucket.py
```

5. Configure S3 bucket as remote backend storage:
```
dvc remote add -d myremote s3://<bucket-name>
```

6. Run Project Pipeline:
```
dvc repro
```

7. Push experiment data to remote backend:
```
dvc push
```

8. Run script: Check S3 bucket for experiment files to confirm push.


## License
This project is licensed under the MIT License. See the LICENSE file for more details.