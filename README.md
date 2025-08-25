# Finite-Differences-and-Floating-Point-Errors
What is done in this project:
1. Floating point arithmetics and errors:
1.1 Demonstrate innacurate calculations by computer due to finite precision
arithmetics.
1.2 For developing test case you can explore round off errors, underflow,
overflow, violation of associative property etc.
2. Finite differences in one and two spatial dimensions:
2.1 Consider finite differences formulas of different precision.
2.2 Visualize errors using tangent line and tangent plane.
2.3 Let discretization step to go to zero and observe behavior of the error.
2.4 Analyze data and draw conclusions.

   
Results of 1:
Round-off error arises because of approximation of square root. We tried same function twice but the second time it was after mathematical manipulations.
f(x,y)= √(x^2+y^2)-x=y^2/(√(x^2+y^2)+x)
As we can see in first case the result is 0.0 while in second case it is 5e-33. This happened because when x=10^16 and y=10^(-8), x^2+y^2≈x^2 because y^2 is negligible in comparison. Thus, in first case  √(x^2+y^2)-x becomes approximately zero and due to the limits of floating-point arithmetic, this results in a computed value of exactly 0. In second case, √(x^2+y^2 )+x≈2x, therefore our function becomes f(x,y)≈y^2/2x and if we put values of x and y in the function, we get (10^(-8))^2/(2*10^16)=5*10^(-33).


Underflow error arises because value of our function becomes lower than the smallest double value (-1.79769e+308).


Overflow error arises because value of our function becomes higher than the largest double value (1.79769e+308).


Associative property violation happens because value of z (which is 1) is so small compared to values of x (which is 1e+16) and y (which is -1e16) that in second case, z doesn’t even get added to y and y just gets approximated to itself which then gets subtracted from x (which is opposite of y) and equals to 0, while in first case x + y happens first which is 0 because they have the same absolute values, just different signs and then z (=1) gets added to 0 which results to 1 and therefore, its small value is no longer lost.


Results of 2:
I have analyzed function: f(x,y)=(sin⁡(xy))/xy (but with condition that if either x = 0 or y = 0, the function equals to 1 because otherwise it would’ve been undefined and caused problem for the code)
I have considered forward difference, backward difference, central difference and Richardson’s extrapolation as well as both d/dx and d/dy.


(the rest can be seen after running the code)
