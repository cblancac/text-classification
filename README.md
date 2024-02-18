# text-classification

In this project, a text classifier has been trained to classify the feeling of a given sentence. Specifically, The categories between which it is possible to classify are: SADNESS, JOY, LOVE, ANGER, FEAR and SURPRISE.


## :gear: Setup
- Clone this repository: `git clone https://github.com/cblancac/text-classification`.
- `pip install -r requirements.txt`.
- To train this model, an instance EC2 (Elastic Compute Cloud) of AWS has been used. More specifically, the type of instance used has been `g4dn.xlarge`, which has allowed us to use GPU.
- An instance with an attached NVIDIA GPU, such as `g4dn.xlarge`, must have the appropriate NVIDIA driver installed. Follow this tutorial to get the appropiate NVIDIA driver: `https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html` (in my case, section Option 3: GRID drivers (G5, G4dn, and G3 instances)

## 	:construction: Data Preparation
The first thing to do in this project is to get a proper dataset. `Hugging Face` has a huge number of datasets, among which is the one used in this project: `emotion`. In the next table, the number of sentences per subset is shown:

| Subset | Size |
| ----- | ---- |
| train | 16.000 |
| test | 2.000 |
| validation | 2.000 |

The dataset is imbalanced; the joy and sadness classes appear frequently, whereas love and surprise are about 5–10 times rarer. 

![image](https://github.com/cblancac/text-classification/assets/105242658/a6374748-0383-4a9a-bf15-1abea622e7b8)

There are several ways to deal with imbalanced data, including:

* Randomly oversample the minority class.
* Randomly undersample the majority class.
* Gather more labeled data from the underrepresented classes.

To keep things simple in this project, we'll work with the raw, unbalanced class frequencies. If you want to learn more about these sampling techniques, we recommend checking out the [Imbalanced-learn library](https://imbalanced-learn.org/stable/). Just make sure that you don't apply sampling methods before creating your train/test splits, or you'll get plenty of leakage between them!


Transformer models have a maximum input sequence length that is referred to as the `maximum context size`. In this project, the model choosen is `distilbert-base-uncased`, which has a maximum context size is 512 tokens. On average, 10 tokens represents 10 words, so our maximum number of words to consider would be ~384, which amounts to a few paragraphs of text. We can get a rough estimate of tweet lengths per emotion by looking at the distribution of words per tweet:

![image](https://github.com/cblancac/text-classification/assets/105242658/4a44ecbc-efe4-4820-9c64-c236137dbfff)

From the plot we see that for each emotion, most tweets are around 15 words long and the longest tweets are well below DistilBERT's maximum context size. Texts that are longer than a model's context size need to be truncated, which can lead to a loss in performance if the truncated text contains crucial information; in this case, it looks like that won't be an issue.



## 	:weight_lifting_man: Training models
Once the dataset is ready, it is time to train the model The model has not been trained from scrach, but a very large pretrained model `distilbert-base-uncased` has been taken from `Hugging Face`, and later on it has been fine-tuned by using the dataset that was just created. At this way, a lot of time and money has been saved, taking the expert knowledge of the pretrained model over of the English language. To train the model, just execute the command `python entry_points/train.py`

![image](https://github.com/cblancac/text-classification/assets/105242658/d67da7af-121a-4c66-8681-3e83b50be20a)


## :tada: Make predictions

Finally the prediction of the feeling of a sentence can be done running the script `python entry_points/train.py`. Executing this script, we will have to introduce our sentence as input and it will give us back a string with the category associated by the model to our sentence (SADNESS, JOY, LOVE, ANGER, FEAR or SURPRISE).

Additionally, an html file will be stored in the `results/` folder, with information on the importance of the 5 most relevant words to make such a decision.

![image](https://github.com/cblancac/text-classification/assets/105242658/42eaf28e-49df-4321-9310-7035e6b24a60)


All the files generated for the text classifier has been uploaded to my `Hugging Face` account, with the `requirements.txt` and the script necessary to run inference using `Gradio`. You could tested directly from my [Hugging Face Spaces](https://huggingface.co/spaces/carblacac/emotion-detection)
![image](https://github.com/cblancac/text-classification/assets/105242658/418547db-1932-41c3-8b7c-49fd91ac1acf)


## :airplane: Deployment

To deploy this model, three services are used from AWS:

1. **EC2**:
The folder lambda-docker need to be storage in the EC2 instance. This folder contains the next files:
  * `Dockerfile`: configuration file used to build a Docker image.
  * `app.py`: Python file containing the code for the application.
  * `distilbert-base-uncased-finetuned-emotion`: directory containing data for a fine-tuned model.
  * `exceptions.py`: Python file that contain definitions of custom exceptions used in the application.
  * `requirements.txt`: file containing a list of Python dependencies required for the application.

Open your terminal, and connect to your virtual machine in AWS. Then you should execute the next commands:
`sudo yum update`, `sudo yum install awscli -y`. Also Docker-compose need to be installed with: `sudo curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose`, `sudo chmod +x /usr/local/bin/docker-compose`, `sudo yum install docker`, `sudo service docker start` and `sudo usermod -a -G docker ec2-user`. Then logout from the instance and login again to verify the installation of Docker.
    
2.  **ECR**:
Once finished with the EC2, you need to go to Amazon Elastic Container Registry. Create a new repository (private) by clicking in the orange button what you see in the image:
![image](https://github.com/cblancac/text-classification/assets/105242658/9d5a20ca-a1af-473f-9d46-cfcd1c9267d3)

Once created, click in your repository name and click in "View push commands". You will see four commands that you need to execute in your ec2, inside of the lambda-docker directory.


3. **Lambda**:
After finish the third step, acces to Lambda service, and click in the "Create function" orange button (see the image).

![image](https://github.com/cblancac/text-classification/assets/105242658/63fe7683-941d-4cf7-95cc-d05a48e814d2)

Then select "Container image" and click "Browse images" to find the image that you have just created (inside of your Amazon ECR image repository). Finished the previous steps, and once the lambda functions is created, click on it, and add a trigger (API Gateway) in the "Configuration" section. The General configuration also need to be edited, as shown in the next picture:

![image](https://github.com/cblancac/text-classification/assets/105242658/017fe051-1f12-4ebc-8afd-369e8e1c961c)


If you complete succesfully all the previous stpes, you are ready to test the text classifier! Just open your Postman, and send a a POST request using the API endpoint from your API Getaway. 

![image](https://github.com/cblancac/text-classification/assets/105242658/b218e67e-1012-4430-8434-37361e1ab556)
