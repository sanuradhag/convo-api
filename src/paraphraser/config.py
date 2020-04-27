import torch

START_TOKEN = '<S>'
END_TOKEN = '<E>'
PAD_TOKEN = '<P>'

CUDA = False

TRAIN_SIZE = 53000
TEST_SIZE = 3000
VALID_SET_SIZE_RATIO = 0.1

dataset_path = '/Users/anuradha/Desktop/convo-api/src/paraphraser/dataset/quora_duplicate_questions.tsv'
emb_path = '//Users/anuradha/Desktop/convo-api/src/paraphraser/dataset/pretrained_word_embeddings/glove_50.npy'
emb_info_path = '/Users/anuradha/Desktop/convo-api/src/paraphraser/dataset/pretrained_word_embeddings/glove_50_info.pkl'

gen_model_path = '/Users/anuradha/Desktop/convo-api/src/paraphraser/model/gen.trc'
dis_model_path = '/Users/anuradha/Desktop/convo-api/src/paraphraser/model/gen.trc'

pretrain_gen_path = '/Users/anuradha/Desktop/convo-api/src/src/paraphraser/model/pretrasin/gen.trc'
pretrain_dis_path = '/Users/anuradha/Desktop/convo-api/src/src/paraphraser/model/pretrain/dis.trc'

MAX_SEQ_LEN_PADDING = 5
BATCH_SIZE = 16
ROLLOUT_NUM = 3

TEACHER_FORCING_RATIO = 0.9
TEACHER_FORCING_RATIO_DECR_STEP = 0.05
TEACHER_FORCING_UPDATE_EP = 5

G_PRETRAIN_EPOCHS =  15
D_PRETRAIN_STEPS =  5
D_PRETRAIN_EPOCHS =  5

G_TRAIN_STEPS =  3
D_TRAIN_STEPS =  3
D_TRAIN_EPOCHS =  2
ADV_TRAIN_ITERS =  15

ED = 50
G_HD = 64
D_HD = 64

model_params = {'gan': {'rn': ROLLOUT_NUM, 'tfr': TEACHER_FORCING_RATIO, 'tfrd': TEACHER_FORCING_RATIO_DECR_STEP,
                        'tfue': TEACHER_FORCING_UPDATE_EP, 'bs': BATCH_SIZE, 'pad': MAX_SEQ_LEN_PADDING},
                'G': {'ed': ED, 'hd': G_HD},
                'D': {'ed': ED, 'hd': D_HD}}

training_params = {'gan': {'iter': ADV_TRAIN_ITERS},
                   'G': {'st': G_TRAIN_STEPS, 'ep': 1},
                   'D': {'st': D_TRAIN_STEPS, 'ep': D_TRAIN_EPOCHS}}

pretrain_params = {'G': {'st': 1, 'ep': G_PRETRAIN_EPOCHS},
                   'D': {'st': D_PRETRAIN_STEPS, 'ep': D_PRETRAIN_EPOCHS}}
