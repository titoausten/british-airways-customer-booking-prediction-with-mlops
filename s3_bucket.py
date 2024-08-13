import boto3


def create_bucket(bucket_name: str):
    s3_obj = boto3.client("s3")
    s3_obj.create_bucket(Bucket = bucket_name)


if __name__ == "__main__":
    bucket_name = input("Enter unique bucket name: ")
    create_bucket(bucket_name=bucket_name)
