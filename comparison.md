# version-4-Temporal Model - 15 frame - train resnet


num_params = 2 M

Without non-maximal supressopn
DATALOADER:0 TEST RESULTS
```yaml
{'input_heatmap_encoder/false negative': 75.90375757575755,
 'input_heatmap_encoder/false positive': 81.31066666666663,
 'input_heatmap_encoder/true positive': 94.09624242424243,
 'input_heatmap_encoder/true positive distance': 15.342394321251655,
 'temporal_encoder/false negative': 133.17337373737374,
 'temporal_encoder/false positive': 17.809090909090905,
 'temporal_encoder/true positive': 36.82662626262629,
 'temporal_encoder/true positive distance': 13.985326246195203}
```
With non-maximal sup

 DATALOADER:0 TEST RESULTS
 ```yaml
{'input_heatmap_encoder/false negative': 75.90375757575755,
 'input_heatmap_encoder/false positive': 81.31066666666663,
 'input_heatmap_encoder/true positive': 94.09624242424243,
 'input_heatmap_encoder/true positive distance': 15.342394321251655,
 'temporal_encoder/false negative': 150.88674747474744,
 'temporal_encoder/false positive': 0.0,
 'temporal_encoder/true positive': 19.113252525252527,
 'temporal_encoder/true positive distance': 16.32333908698632}
```

# version-6-Temporal Model - 5 frame - train resnet
num_params = 991k
DATALOADER:0 TEST RESULTS
```yaml
{'input_heatmap_encoder/false negative': 90.66646464646465,
 'input_heatmap_encoder/false positive': 29.56004040404042,
 'input_heatmap_encoder/true positive': 79.33353535353531,
 'input_heatmap_encoder/true positive distance': 14.293738602625222,
 'temporal_encoder/false negative': 121.91785858585857,
 'temporal_encoder/false positive': 0.0,
 'temporal_encoder/true positive': 48.08214141414144,
 'temporal_encoder/true positive distance': 2.7763857164186434}
```

# version-7-Temporal Model - 3 frame - train resnet
param = 991 K
DATALOADER:0 TEST RESULTS
```yaml
{'input_heatmap_encoder/false negative': 54.14403966808333,
 'input_heatmap_encoder/false positive': 118.82210078931392,
 'input_heatmap_encoder/true positive': 115.8559603319167,
 'input_heatmap_encoder/true positive distance': 14.885247260888201,
 'temporal_encoder/false negative': 126.63934426229503,
 'temporal_encoder/false positive': 0.0,
 'temporal_encoder/true positive': 43.36065573770489,
 'temporal_encoder/true positive distance': 2.20659483240178}
 ```
# version-5-Temporal Model - 1 frame - train resnet

num_params = 2 M

DATALOADER:0 TEST RESULTS
```yaml
{'input_heatmap_encoder/false negative': 91.3860436893204,
 'input_heatmap_encoder/false positive': 9.202305825242721,
 'input_heatmap_encoder/true positive': 78.61395631067963,
 'input_heatmap_encoder/true positive distance': 13.59089603805996,
 'temporal_encoder/false negative': 120.46545307443355,
 'temporal_encoder/false positive': 0.0,
 'temporal_encoder/true positive': 49.534546925566325,
 'temporal_encoder/true positive distance': 2.209062168507912}
```

 # version-8-Temporal Model - 1 frame - train resnet
param = 937 K
DATALOADER:0 TEST RESULTS
```yaml
{'input_heatmap_encoder/false negative': 84.58021466180637,
 'input_heatmap_encoder/false positive': 13.502126366950183,
 'input_heatmap_encoder/true positive': 85.4197853381936,
 'input_heatmap_encoder/true positive distance': 14.286724190189481,
 'temporal_encoder/false negative': 120.13950992304588,
 'temporal_encoder/false positive': 0.0,
 'temporal_encoder/true positive': 49.860490076954285,
 'temporal_encoder/true positive distance': 2.3121277821546453}


 ```

# version-9-Temporal Model - 3 frame - fix resnet

# version-10-Temporal Model -removed no batch norm at output - 3 frame - fix resnet
```yaml
{'input_heatmap_encoder/false negative': 155.11627200971478,
 'input_heatmap_encoder/false positive': 0.0,
 'input_heatmap_encoder/true positive': 14.883727990285369,
 'input_heatmap_encoder/true positive distance': 38.43949741496634,
 'temporal_encoder/false negative': 133.47269783444645,
 'temporal_encoder/false positive': 0.0,
 'temporal_encoder/true positive': 36.527302165553536,
 'temporal_encoder/true positive distance': 38.76044630868471}
```