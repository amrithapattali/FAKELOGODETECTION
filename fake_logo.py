import requests
from bs4 import BeautifulSoup
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tensorflow.keras.applications import VGG16
import easyocr

def download_image(url, keyword, index):
    """
    Downloads an image from the given URL and saves it with a specific filename.
    
    Args:
    - url (str): The URL of the image to download.
    - keyword (str): The keyword associated with the image.
    - index (int): The index of the image in the search results.
    
    Returns:
    - str or None: The filename of the downloaded image, or None if download fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        
        if response.status_code == 200:
            filename = f"{keyword}_{index}.jpg"
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while downloading image: {str(e)}")
    return None

def search_and_download_images(keyword, max_images=10):
    """
    Searches Google Images for images related to the given keyword and downloads them.
    
    Args:
    - keyword (str): The keyword to search for.
    - max_images (int): The maximum number of images to download.
    
    Returns:
    - list[str] or None: List of filenames of downloaded images, or None if search fails.
    """
    try:
        search_query = f"https://www.google.com/search?q={keyword}&tbm=isch"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(search_query, headers=headers)
        response.raise_for_status()  # Raise an exception for non-200 status codes

        soup = BeautifulSoup(response.content, 'html.parser')
        img_tags = soup.find_all('img')
        img_urls = [img['src'] for img in img_tags if img.get('src') and img['src'].startswith('http')]

        downloaded_images = []
        for i, img_url in enumerate(img_urls[:max_images]):
            image_path = download_image(img_url, keyword, i + 1)
            if image_path:
                downloaded_images.append(image_path)
        return downloaded_images
    except Exception as e:
        print(f"Error occurred while searching and downloading: {str(e)}")
        return None

def load_and_preprocess_image(image_path):
    """
    Loads and preprocesses an image from the given file path.
    
    Args:
    - image_path (str): The file path of the image.
    
    Returns:
    - numpy.ndarray: Preprocessed image array.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array

def extract_features(image_path):
    """
    Extracts features from an image using the VGG16 model.
    
    Args:
    - image_path (str): The file path of the image.
    
    Returns:
    - numpy.ndarray: Feature vector extracted from the image.
    """
    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    img = load_and_preprocess_image(image_path)
    features = model.predict(img)
    return features.flatten()

def cosine_similarity(features1, features2):
    """
    Calculates the cosine similarity between two feature vectors.
    
    Args:
    - features1 (numpy.ndarray): Feature vector 1.
    - features2 (numpy.ndarray): Feature vector 2.
    
    Returns:
    - float: Cosine similarity score.
    """
    dot_product = np.dot(features1, features2)
    norm_features1 = np.linalg.norm(features1)
    norm_features2 = np.linalg.norm(features2)
    similarity = dot_product / (norm_features1 * norm_features2)
    return similarity

def pearson_correlation(features1, features2):
    """
    Calculates the Pearson correlation coefficient between two feature vectors.
    
    Args:
    - features1 (numpy.ndarray): Feature vector 1.
    - features2 (numpy.ndarray): Feature vector 2.
    
    Returns:
    - float: Pearson correlation coefficient.
    """
    correlation_coefficient, _ = pearsonr(features1, features2)
    return correlation_coefficient

def extract_text(image_path):
    """
    Extracts text from an image using EasyOCR.
    
    Args:
    - image_path (str): The file path of the image.
    
    Returns:
    - str: Extracted text from the image.
    """
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    text = ' '.join([result[1] for result in results]) if results else ''
    return text

def calculate_image_similarity(image1_path, image2_path):
    """
    Calculates the similarity between two images based on features and text.
    
    Args:
    - image1_path (str): The file path of the first image.
    - image2_path (str): The file path of the second image.
    
    Returns:
    - float: Combined similarity score.
    """
    text_similarity = 1 if extract_text(image1_path) == extract_text(image2_path) else -1 if extract_text(
        image1_path) and extract_text(image2_path) else 0
    features1 = extract_features(image1_path)
    features2 = extract_features(image2_path)
    cosine_sim = cosine_similarity(features1, features2)
    pearson_corr = pearson_correlation(features1, features2)
    combined_similarity = (0.4 * cosine_sim) + (0.3 * (pearson_corr + 1) / 2)
    if text_similarity == 1:
        combined_similarity += 1
    elif text_similarity == -1:
        combined_similarity -= 1
    return combined_similarity

if __name__ == "__main__":
    # Accept input for brand logo name and test image path
    keyword = input("Enter the brand with category name: ")
    test_image_path = input("Enter the path to the test image: ")

    # Extract text from the test image
    test_text = extract_text(test_image_path)

    # Search and download images related to the keyword
    real_image_paths = search_and_download_images(keyword)
    
    # If real images are found
    if real_image_paths:
        similarity_scores = []

        # Calculate similarity scores for each real image
        for real_image_path in real_image_paths:
            real_text = extract_text(real_image_path)
            similarity_score = calculate_image_similarity(test_image_path, real_image_path)
            similarity_scores.append((similarity_score, real_image_path))

        # Sort similarity scores in descending order
        similarity_scores.sort(reverse=True)

        # If there are similarity scores
        if similarity_scores:
            max_similarity_score, _ = similarity_scores[0]

            # Determine prediction based on max similarity score
            if max_similarity_score > 0.75:
                prediction = "The logo is Real...!!!"
            else:
                prediction = "The logo is fake..!!!"
        else:
            prediction = "No similar logo  found."

        # Display the test image with prediction
        plt.figure(figsize=(8, 6))
        plt.imshow(plt.imread(test_image_path))
        plt.title(prediction, fontsize=16, color='Black', loc='center', pad=20)
        plt.axis('off')
        plt.show()
    else:
        print("No images found for the given keyword.")

    # Function to delete downloaded images
def delete_downloaded_images(image_paths):
    for image_path in image_paths:
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image: {image_path}")
        else:
            print(f"Image not found: {image_path}")

downloaded_images = search_and_download_images(keyword, max_images=10)

# Delete downloaded images after use
delete_downloaded_images(downloaded_images)
