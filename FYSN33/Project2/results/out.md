## Printed results
Terminal output corresponding to the figures in `results/`:

```bash
Sequential(
  (0): Linear(in_features=10, out_features=32, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.1, inplace=False)
  (3): Linear(in_features=32, out_features=16, bias=True)
  (4): ReLU()
  (5): Linear(in_features=16, out_features=1, bias=True)
)
Number of trainable parameters: 897
Epoch 001        | Train Loss: 0.6840    | Val Loss: 0.6718      | Train Acc: 0.542 | Val Acc: 0.564
Epoch 002        | Train Loss: 0.6623    | Val Loss: 0.6528      | Train Acc: 0.566 | Val Acc: 0.581
Epoch 003        | Train Loss: 0.6473    | Val Loss: 0.6401      | Train Acc: 0.581 | Val Acc: 0.582
Epoch 004        | Train Loss: 0.6393    | Val Loss: 0.6315      | Train Acc: 0.594 | Val Acc: 0.590
Epoch 005        | Train Loss: 0.6339    | Val Loss: 0.6269      | Train Acc: 0.600 | Val Acc: 0.594
Epoch 006        | Train Loss: 0.6287    | Val Loss: 0.6244      | Train Acc: 0.602 | Val Acc: 0.588
Epoch 007        | Train Loss: 0.6297    | Val Loss: 0.6223      | Train Acc: 0.603 | Val Acc: 0.592
Epoch 008        | Train Loss: 0.6272    | Val Loss: 0.6213      | Train Acc: 0.607 | Val Acc: 0.598
Epoch 009        | Train Loss: 0.6257    | Val Loss: 0.6208      | Train Acc: 0.607 | Val Acc: 0.597
Epoch 010        | Train Loss: 0.6262    | Val Loss: 0.6198      | Train Acc: 0.611 | Val Acc: 0.596
Epoch 011        | Train Loss: 0.6243    | Val Loss: 0.6187      | Train Acc: 0.611 | Val Acc: 0.598
Epoch 012        | Train Loss: 0.6221    | Val Loss: 0.6177      | Train Acc: 0.617 | Val Acc: 0.607
Epoch 013        | Train Loss: 0.6234    | Val Loss: 0.6186      | Train Acc: 0.618 | Val Acc: 0.610
Epoch 014        | Train Loss: 0.6181    | Val Loss: 0.6181      | Train Acc: 0.617 | Val Acc: 0.604
Epoch 015        | Train Loss: 0.6201    | Val Loss: 0.6166      | Train Acc: 0.623 | Val Acc: 0.613
Epoch 016        | Train Loss: 0.6203    | Val Loss: 0.6166      | Train Acc: 0.625 | Val Acc: 0.606
Epoch 017        | Train Loss: 0.6151    | Val Loss: 0.6170      | Train Acc: 0.627 | Val Acc: 0.618
Epoch 018        | Train Loss: 0.6146    | Val Loss: 0.6159      | Train Acc: 0.627 | Val Acc: 0.612
Epoch 019        | Train Loss: 0.6162    | Val Loss: 0.6157      | Train Acc: 0.631 | Val Acc: 0.614
Epoch 020        | Train Loss: 0.6162    | Val Loss: 0.6155      | Train Acc: 0.632 | Val Acc: 0.617
Epoch 021        | Train Loss: 0.6183    | Val Loss: 0.6150      | Train Acc: 0.631 | Val Acc: 0.614
Epoch 022        | Train Loss: 0.6168    | Val Loss: 0.6149      | Train Acc: 0.621 | Val Acc: 0.617
Epoch 023        | Train Loss: 0.6176    | Val Loss: 0.6147      | Train Acc: 0.626 | Val Acc: 0.620
Epoch 024        | Train Loss: 0.6185    | Val Loss: 0.6151      | Train Acc: 0.626 | Val Acc: 0.620
Epoch 025        | Train Loss: 0.6133    | Val Loss: 0.6142      | Train Acc: 0.629 | Val Acc: 0.618
Epoch 026        | Train Loss: 0.6157    | Val Loss: 0.6149      | Train Acc: 0.625 | Val Acc: 0.616
Epoch 027        | Train Loss: 0.6099    | Val Loss: 0.6137      | Train Acc: 0.632 | Val Acc: 0.623
Epoch 028        | Train Loss: 0.6152    | Val Loss: 0.6133      | Train Acc: 0.627 | Val Acc: 0.626
Epoch 029        | Train Loss: 0.6142    | Val Loss: 0.6136      | Train Acc: 0.634 | Val Acc: 0.619
Epoch 030        | Train Loss: 0.6092    | Val Loss: 0.6143      | Train Acc: 0.634 | Val Acc: 0.614
Epoch 031        | Train Loss: 0.6162    | Val Loss: 0.6154      | Train Acc: 0.627 | Val Acc: 0.613
Epoch 032        | Train Loss: 0.6110    | Val Loss: 0.6135      | Train Acc: 0.639 | Val Acc: 0.611
Epoch 033        | Train Loss: 0.6164    | Val Loss: 0.6140      | Train Acc: 0.634 | Val Acc: 0.620
Epoch 034        | Train Loss: 0.6103    | Val Loss: 0.6138      | Train Acc: 0.638 | Val Acc: 0.615
Epoch 035        | Train Loss: 0.6103    | Val Loss: 0.6135      | Train Acc: 0.628 | Val Acc: 0.617
Epoch 036        | Train Loss: 0.6149    | Val Loss: 0.6173      | Train Acc: 0.634 | Val Acc: 0.614
Epoch 037        | Train Loss: 0.6134    | Val Loss: 0.6136      | Train Acc: 0.635 | Val Acc: 0.610
Epoch 038        | Train Loss: 0.6109    | Val Loss: 0.6140      | Train Acc: 0.632 | Val Acc: 0.609
Early stopping triggered.
Validation AUC: 0.6817274793857151

Chosen working point (optimal threshold), t* = 0.6064
Purity P(t*) = 0.6608
Signal efficiency epsilon_S = 0.3026
Background efficiency epsilon_B = 0.1373

Baseline purity (no cuts) = 0.476  with S0=3135, B0=3447
MC (cuts): purity S/(S+B) = 0.526  with S=1735, B=1566, N=3301  |  eps_S=0.553, eps_B=0.454
MC (NN): purity S/(S+B) = 0.688  with S=969, B=440, N=1409  |  eps_S=0.309, eps_B=0.128

Data (cuts): slected = 46.799 %
Data (NN): slected = 13.847 %
```