import zipfile

filenames = ["./part1/info.yaml",
             "./part1/FruitModel.py",
             './part1/model/attention.npy',
             './part1/autograd/BaseNode.py',
             './part1/autograd/BaseGraph.py',
             './part2/LLM.py',
             './part2/log.txt',
             './part2/res_en.txt',
             './part2/res.txt']

f = zipfile.ZipFile("answer.zip", "w", zipfile.ZIP_DEFLATED)
for filename in filenames:
    f.write(filename)
f.close()