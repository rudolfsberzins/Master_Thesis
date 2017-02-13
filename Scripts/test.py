from ppi_predict import PPIclassifier as ppi

CLF = ppi()

FILES = ["../MEDLINE_FILES/", "../Text_Mining_files/output_mentions_yeast_1",
         "../Text_Mining_files/output_pairs_yeast_1",
         "../Text_Mining_files/yeast/yeast_entities.tsv",
         "../Text_Mining_files/4932.protein.actions.v10.txt/4932.protein.actions.v10.txt"]
MENTS, PAIRS, ENTS, INTERS = CLF.setup(FILES)
TRAIN_SET, TEST_SET = CLF.produce_dfs(100, 0.3)
ERROR, PROBS = CLF.predict(1000, 100, 23)

print ERROR

print "\nDONE WITH THE FIRST ONE!\n"

ERROR2, PROBS2 = CLF.predict(1000, 100, 21, train=TRAIN_SET, test=TEST_SET)

print ERROR2
