# Deep Learning 101 [DL-101]
## Everything you need to learn to train Deep Neural Nets 

The repository holds the implementation of technical concepts involved the Deep Learning domain. We opt a practical approach of learning where we pick a Dataset and start training a Deep Neural Network on it and incrementally implementing optimization techniques and best-practices and understanding it's impact on the Prediction and Loss.

## Understanding the Problem Statement

We pick up a competition that runs indefinitely on Kaggle named [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview). Let's understand more about the challenge.

The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.

While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!

<img src="/media/competition.jpg" width=600 height=500>

## Understanding the Dataset

| S.No. | Field | Description |
|---|---|--|
|1. | PassengerId | A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is traveling with and pp is their number within the group. People in a group are often family members, but not always.|
|2. | HomePlanet | The planet the passenger departed from, typically their planet of permanent residence.|
|3. | CryoSleep  | Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.|
|4. | Cabin | The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.|
|5. | Destination | The planet the passenger will be debarking to.|
|6. | Age | The age of the passenger|
|7. | VIP | Whether the passenger has paid for special VIP service during the voyage.|
|8. |RoomService, FoodCourt, ShoppingMall, Spa, VRDeck | Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.|
|13. |Name | The first and last names of the passenger|
|14. |Transported | Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.|

```
- Training Data: Personal records for about two-thirds (~8700) of the passengers, to be used as training data
- Test Data: Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. 
- Goal: Your task is to predict the value of Transported for the passengers in this set.
```

## Topics Covered

1. Building your Deep Neural Net architecture.
2. Training your Neural Net
3. Evaluating Dev and Test Losses
4. L2 Regularization
5. Drop-out Regularization
6. Vanishing / Exploding Gradients
7. Gradient Descent with Momentum
8. RMS Prop
9. Adam Optimization 
10. Batch Normalization (Batch Norm)
11. Early Stopping
12. Conclusions