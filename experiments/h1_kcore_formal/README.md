# Checkpoint compatibility for the formal k-core experiments

The formal trials used k=3, t<=0.5 and a feature threshold of 0.20.
The feature path and the final H1 edge decision use separate thresholds.
The final edge threshold and HOG remain at 0.50.
The standalone preset does not select the formal threshold by itself.

Use PilotH1 in pilot_model.py for the SBM pilot/formal500 architecture,
or get_h1_class(dataset) in model_runtime.py for the collection/large-graph
transfer arms. The Python search path should include this directory and
dgd (and the applicable project helper directories). These are the model
adapters; using exact frozen inputs and running the server workflow
requires the matching release bundle and experiment artifacts.
