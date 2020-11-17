
# W/O training input encoder

### Temporal Encoder
### version - 0 - 1 frame

- params: 3 M
- Type: Unet
- depth: 4

```javascript
{'input_heatmap_encoder/false negative': 85.4259394102841,
 'input_heatmap_encoder/false positive': 0.0008899676375404531,
 'input_heatmap_encoder/true positive': 14.479944264653003,
 'input_heatmap_encoder/true positive distance': 34.00602625749318,
 'temporal_encoder/false negative': 54.41196961524632,
 'temporal_encoder/false positive': 0.02131877022653724,
 'temporal_encoder/true positive': 45.49391405969074,
 'temporal_encoder/true positive distance': 33.69700338313878}
```

### version - 1&3 - 3 frame 

training with 3 epoch and 12 epoch have same result

- params: 3 M
- Type: Unet
- depth: 4

```javascript
{'input_heatmap_encoder/false negative': 86.26990487755516,
 'input_heatmap_encoder/false positive': 0.0012952843553936451,
 'input_heatmap_encoder/true positive': 14.882432705929977,
 'input_heatmap_encoder/true positive distance': 40.069497334451405,
 'temporal_encoder/false negative': 51.626431896377284,
 'temporal_encoder/false positive': 0.10232746407609826,
 'temporal_encoder/true positive': 49.52590568710784,
 'temporal_encoder/true positive distance': 40.661609002917416}
```

### version - 5 - 5 frame

- params: 3 M
- Type: Unet
- depth: 4

```javascript
{'input_heatmap_encoder/false negative': 86.65595959595969,
 'input_heatmap_encoder/false positive': 0.00101010101010101,
 'input_heatmap_encoder/true positive': 14.736525252525249,
 'input_heatmap_encoder/true positive distance': 41.24996667073625,
 'temporal_encoder/false negative': 57.02307070707068,
 'temporal_encoder/false positive': 0.03309090909090912,
 'temporal_encoder/true positive': 44.36941414141414,
 'temporal_encoder/true positive distance': 41.7528729063837}
```

### version - 6 - 7 frame

- params: 4 M
- Type: Unet
- depth: 4

```javascript
{'input_heatmap_encoder/false negative': 86.84891041162228,
 'input_heatmap_encoder/false positive': 0.0008878127522195318,
 'input_heatmap_encoder/true positive': 14.664689265536731,
 'input_heatmap_encoder/true positive distance': 41.758735730812184,
 'temporal_encoder/false negative': 51.07816787732038,
 'temporal_encoder/false positive': 0.05726392251815972,
 'temporal_encoder/true positive': 50.43543179983856,
 'temporal_encoder/true positive distance': 42.68813669026329}

```

### version - 6 - 9 frame

- params: 4 M
- Type: Unet
- depth: 4

```javascript
{}

```


<!-- # version-4-Temporal Model - 15 frame - train resnet


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
 'temporal_encoder/true positive distance': 38.76044630868471} -->
```