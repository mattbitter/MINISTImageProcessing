# MINISTImageProcessing

Raw data not included.

Created 20x20 pixel stretched bounded boxes of all digits.

Ran it through Gaussian and Bernoulli Naive Bayes

As well as dicision forests of varing lengths.

Python 3.6 was used to code, PyCharm was used for the IDE, Alt+Shift+E was used to run the

code in sections. It was interesting that the Gaussian and the Bernoulli for NB flipped their accuracy

depending on the formatting of the image data. I believe this was because the Bernoulli is meant for

binary features. Specifically, the untouched images hard more white spaces which the Bernoulli

distribution could clearly mark as &#39;0&#39; while the Gaussian distribution was forced to still mark the 0&#39;s

following a normal distribution which was not accurate. When the images were bounded, the Gaussian

performed better because there was less white space and more ink for the normal distribution to model.

For my Bounded stretched box, I choose to keep the grey scale in the cropped and stretched image. Also I used

the mean threshold per image as the cropping threshold rather than making it a constant for all images

because it would be more flexible for lighter drawn numbers.

| Accuracy (F1 Score) | Gaussian | Bernoulli |
| --- | --- | --- |
| untouched images         | 52%     | 83% |
| stretched bounding box   | 83%     | 76% |

Untouched raw pixels:

| Accuracy (5-fold CV mean) | depth = 4 | depth = 8 | depth = 16 |
| --- | --- | --- | --- |
| #trees = 10                 | 75% | 89% | 93% |
| #trees = 20                     | 91% | 95% | 78% |
| #trees = 30                     | 91% | 95% | 79% |

Stretched bounding box:

| Accuracy (5-fold CV mean) | depth = 4 | depth = 8 | depth = 16 |
| --- | --- | --- | --- |
| #trees = 10                     | 74% | 90% | 94% |
| #trees = 20                     |  77% |  92% | 95% |
| #trees = 30                     |  78% |  92% | 96% |
