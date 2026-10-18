import azure.functions as func


def main(inputblob: func.InputStream) -> None:
    from processor import process_raw_image

    process_raw_image(inputblob)
