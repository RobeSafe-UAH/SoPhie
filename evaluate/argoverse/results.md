## Problems
* dummies
    * bad influence in model
* Normalization of rel and abs
* Imbalanced "class" problem
    * relatives min and max
    * few curves in dataset (?)
* Model output in range (0.4,0.6). Normalized
* Decoder Activation Function
    * Sigmoid
    * ReLU
    * PReLU
* Temporal Decoder
    * It takes into account full trajectory
* Layernorm and LeakyReLU 
* Train
    * Generator -> Normalized
    * Discriminator -> no Normalized
* Deep Prediction model based on sgan
    * abs normalized -> min 0
    * rel normalized -> calculated with abs norm
* Trainer problem
    * discriminator step -> detach()
    * check_accuracy -> mask

* multi head self attention local
    * mask concentrate information in agent
* learning rate scheduler
* weight decay
* l2 weight
    * [1, 0.25, 0.2, 0.15, 0.1 , 0.05]

## Experimento 1
* multi to single
* Decoder
* only social attention
* rel trajectory space (-3.5,3.5)
* metrics
    * 6.503
    * 14.565

## Experimento 2
* 1400
    * bce loss
    * ade: 12.6
    * fde: 24.7
* 1400
    * manual bce loss - no sigmoid discriminator
    * ade: 12.7
    * fde: 24.8