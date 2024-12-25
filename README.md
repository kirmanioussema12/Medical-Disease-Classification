# Medical-Disease-Classification
Project Overview
This project develops a text classification model using BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art model for natural language processing (NLP). The goal is to classify medical research papers into predefined categories based on their content. The dataset consists of a collection of medical research papers, each labeled with a specific category related to a particular medical field such as diseases, treatment types, or medical methodologies.

The classification model is trained to understand the complexities of medical language, including terminology and structure, to categorize papers accurately. This project is built using Python, Hugging Face's Transformers library, and PyTorch, and involves several key steps: data loading, preprocessing, tokenization, model training, and evaluation.


Dataset
The dataset used in this project contains medical research papers. Each paper is labeled with a category that represents the main subject of the paper, such as specific diseases, medical specialties, or treatment methods. The dataset is used to train a text classification model to predict the category of a new, unseen paper based on its abstract and title.

Features of the dataset:

Text: Contains the body of medical research papers, including titles, abstracts, and other textual information.
Labels: Each paper is labeled with a specific category that helps to classify the paper into predefined groups.



Steps Involved in the Project
The project follows these key steps to build a successful text classification model:

1. Data Loading
The dataset consists of multiple JSON files, each containing information about medical papers. These files are loaded into a single Pandas DataFrame for ease of processing. The loading process involves:

Reading JSON Files: Using the pandas.read_json() function to load each JSON file containing the paper data.
Merging Data: All JSON files are merged into one DataFrame, where each row corresponds to one paper with its associated title, abstract, and category label.

2. Data Preprocessing
Once the data is loaded into a single DataFrame, it is important to preprocess the text data before feeding it into the model. This step involves:

Cleaning Text: Removing unnecessary characters, special symbols, and irrelevant information from the text.
Tokenization: Splitting the text into tokens (words or subwords) that the model can understand. BERT uses subword tokenization, which ensures that even unseen words can be processed by breaking them down into smaller, known parts.
Label Encoding: The target labels (categories) are encoded numerically to be used in the training process.


3. Tokenization and Data Loading
For BERT to process the text, the data must be tokenized using a BERT tokenizer. This converts the text into numerical representations that BERT can understand. The steps involved are:

Tokenization: Using BertTokenizer to split the text into tokens and convert them into input IDs. The text is padded to ensure that all sequences have the same length.
Data Splitting: The dataset is split into training and validation sets to ensure that the model can be properly evaluated during training.




4. Model Training
The core of the project is training a BERT model for sequence classification. Key steps in this phase include:

Model Selection: Using a pre-trained BERT model (bert-base-uncased) and fine-tuning it for the specific medical text classification task.
Fine-Tuning: Adjusting the model's weights through backpropagation based on the loss between predicted and actual labels.
Training Parameters: The model is trained using specific hyperparameters such as batch size, number of epochs, and learning rate.



5. Evaluation and Metrics
After training, the model is evaluated on a validation or test set using various metrics:

Accuracy: Measures the percentage of correct classifications.
Precision: Measures the proportion of true positives out of all positive predictions.
Recall: Measures the proportion of true positives out of all actual positives.
F1-Score: The harmonic mean of precision and recall, providing a single measure of model performance.


6. Addressing Class Imbalance
The dataset may suffer from class imbalance, where certain categories are underrepresented. To tackle this:

Class Weights: Weights are applied to the loss function to give more importance to the minority classes.
Resampling: Techniques such as oversampling or undersampling may also be considered to balance the class distribution.




Running the Project
Prerequisites
To run this project, you need to have Python installed along with the following libraries:

transformers
torch
sklearn
pandas
numpy






Conclusion
This project demonstrates the power of BERT in the domain of medical text classification. By utilizing the pre-trained BERT model and fine-tuning it on medical papers, we can create a model capable of classifying papers into relevant categories. Future work could include optimizing the model for class imbalance, hyperparameter tuning, and exploring other advanced techniques for fine-tuning BERT models.


