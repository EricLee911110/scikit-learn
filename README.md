# scikit-learn
The first thing we're doing here is to make sure we know how the "plt" works. 

In the following picture, we can see that the data are identical, but the color of the class got messed up.

![image](https://user-images.githubusercontent.com/79837982/167056356-6e183ffa-8a6e-4be6-bb4f-fc3918e455ef.png)

Why you need to [standardize your data before fitting a ML model](https://towardsdatascience.com/how-and-why-to-standardize-your-data-996926c2c832)? 
It is because variables are measured at different scale. They might end up creating a bias.

In this example you might think the "Number of Households" isn't important. Every data is about the same, there is no point analyzing it. (Left)

But it turns out that people have medium number of households tends to have higher income. (Right)

![image](https://user-images.githubusercontent.com/79837982/167063589-9665d33a-7679-4e81-8150-a7cba9d30014.png)

# Classifier Comparision
![image](https://user-images.githubusercontent.com/79837982/167102180-15787a5a-d265-489a-8ac1-0488601871c5.png)
