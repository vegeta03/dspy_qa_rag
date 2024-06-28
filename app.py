import dspy
from dspy.datasets import HotPotQA
from rich import print

# 1. Setup the LM, Configuration and Data Loading
llama3_simpo = dspy.OllamaLocal(
    model="r3m8/llama3-simpo:8b-instruct-q3_K_M",
    model_type="chat",
    max_tokens=2048,
    num_ctx=8000,
    temperature=0.0,
    top_p=0.9,
    top_k=40,
    frequency_penalty=0.1,
    presence_penalty=0.0,
    n=3,
)
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=llama3_simpo, rm=colbertv2_wiki17_abstracts)
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=20)

trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

print(len(trainset), len(devset))
print(f"Trainset Data {trainset[:5]}")
print(f"Devset Data {devset[:5]}")

print("\n### Example Question with Answer ###\n")
example = devset[18]
print(f"Question: {example.question}")
print(f"Answer: {example.answer}")
print(f"Relevant Wikipedia Titles: {example.gold_titles}")

# 2. Basic Chatbot
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 nad 5 words")

print("\n### Generate Response ###\n")
generate_answer = dspy.Predict(BasicQA)
prediction = generate_answer(question=example.question)
print(f"Question: {example.question} \nPredicted Answer: {prediction.answer}")
