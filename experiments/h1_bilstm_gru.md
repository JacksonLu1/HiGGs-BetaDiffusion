# H1 BiLSTM, two-layer MLP and GRU

Choose `model=beta_h1_twohop_bilstm_hog_rw_gru` for the prototype.
H1 builds one symmetric token per valid intermediate node of each unordered
node pair. It sorts those tokens lexicographically by features, then
concatenates the two final BiLSTM hidden states. The existing two-layer MLP
and direct GRU edge update consume the resulting message. There is one GRU
update per pair and no recurrent state across diffusion steps.

The complete set of valid intermediate nodes is used. A valid pair with no
intermediate nodes gets a zero BiLSTM summary. The edge output is symmetric
and invalid pairs stay masked. The BiLSTM needs work proportional to the
number of valid triples, unlike the previous batched matrix product. Pair
chunking and activation checkpointing control memory use during training.

The MLP final layer still starts at zero; the GRU is not identity-initialized.
The checkpoint uses metadata version 5 and should be trained from scratch.
The architecture retains the HOG RW/SPD inputs and Beta diffusion schedule.

Prior source tags: `h1-mean-twohop-hog-sigmoid-v1`, `h1-kcore-encoding-v1`,
`h1-mean-twohop-hog-gru-v1`.
