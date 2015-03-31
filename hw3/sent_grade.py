import models,sys
filename='data/lm'
lm = models.LM(filename)
sentence = "This is a test ."
lm_state = lm.begin() # initial state is always <s>
logprob = 0.0
line = sys.stdin.readlines()
for word in line[0].split():
    (lm_state, word_logprob) = lm.score(lm_state, word)
    logprob += word_logprob
logprob += lm.end(lm_state) # transition to </s>, can also use lm.score(lm_state, "</s>")[1]
print logprob
