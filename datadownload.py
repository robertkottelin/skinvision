isic image download images/  # downloads the entire archive, images and metadata, to images/

# optionally filter the results
isic image download --search 'diagnosis:"basal cell carcinoma"' images/
isic image download --search 'age_approx:[5 TO 25] AND sex:male' images/