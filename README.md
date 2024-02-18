# text-classification

In this project, a text classifier has been trained to classify the feeling of a given sentence. Specifically, The categories between which it is possible to classify are: SADNESS, JOY, LOVE, ANGER, FEAR and SURPRISE.


## :gear: Setup
- Clone this repository: `git clone https://github.com/cblancac/text-classification`.
- `pip install -r requirements.txt`.
- To train this model, an instance EC2 (Elastic Compute Cloud) of AWS has been used. More specifically, the type of instance used has been `g4dn.xlarge`, which has allowed us to use GPU.
- An instance with an attached NVIDIA GPU, such as `g4dn.xlarge`, must have the appropriate NVIDIA driver installed. Follow this tutorial to get the appropiate NVIDIA driver: `https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html` (in my case, section Option 3: GRID drivers (G5, G4dn, and G3 instances)

## 	:construction: Data Preparation
The first thing to do in this project is to get a proper dataset. Hugging Face has a huge number of datasets, among which is the one used in this project: `emotion`. In the next table, the number of sentences per subset is shown:

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


