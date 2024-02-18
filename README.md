# text-classification

In this project, a text classifier has been trained to classify the feeling of a given sentence. Specifically, The categories between which it is possible to classify are: SADNESS, JOY, LOVE, ANGER, FEAR and SURPRISE.


## :gear: Setup
- Clone this repository: `git clone https://github.com/cblancac/text-classification`.
- `pip install requirements.txt`.
- To train this model, an instance EC2 (Elastic Compute Cloud) of AWS has been used. More specifically, the type of instance used has been `g4dn.xlarge`, which has allowed us to use GPU.
- An instance with an attached NVIDIA GPU, such as `g4dn.xlarge`, must have the appropriate NVIDIA driver installed. Follow this tutorial to get the appropiate NVIDIA driver: `https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html` (in my case, section Option 3: GRID drivers (G5, G4dn, and G3 instances)

## 	:construction: Data Preparation
The first thing to do in this project is to get a proper dataset. Hugging Face has a huge number of datasets, among which is the one used in this project: `emotion`. In the next table, the number of sentences per subset is shown:

| Subset | Size |
| ----- | ---- |
| train | 16.000 |
| test | 2.000 |
| validation | 2.000 |
