## Sustainable Portfolio Management Bachelor Thesis
This repository contains the code and report for a bachelor thesis on the efficient frontier and ESG in portfolio optimization.

### Introduction
In this thesis, we explore the efficient frontier using modern portfolio theory and different optimization techniques. The efficient frontier is a concept in finance that represents the set of portfolios that offer the highest expected return for a given level of risk or the lowest risk for a given level of expected return. The efficient frontier is an important tool in portfolio optimization, as it helps investors to find the optimal portfolio that balances risk and return.

### Data Library
To access the data and run the efficient frontier for ESG data implementation, please run the function get_filtered_stock_data() from the file DATA.py and run it in MAIN.ipynb. The get_filtered_stock_data() function is used to filter the stock data based on different criteria, which you can read more about in the file DATA.py.

### Efficient Frontier Library
The efficient_frontier.py module contains utility functions for portfolio optimization, including the construction of the efficient frontier using ESG data. You can also run this module in Efficient_frontier.ipynb.

### Backtest Library
The backtest.py module is a comprehensive library that provides various functions for conducting backtesting of investment strategies. It includes features such as data retrieval, strategy implementation, performance evaluation, and risk analysis. To utilize this library, simply import the module and call the desired functions in your code.

### Greenwashing Library
The Greenwashing Library provides a portfolio optimization algorithm specifically designed to tackle greenwashing practices.

### User Features Library
The user_features.py module is a versatile library designed to enhance user experience in financial applications. It includes functions for showcasing optimal weight campaigns and printing function comments. These features facilitate interaction, provide insights, and improve the usability of the library. By leveraging the user_features.py module, developers can create user-friendly and interactive financial applications.

### Ipynb
In separate Jupyter Notebook files, different libraries are utilized to address distinct objectives. Each notebook focuses on specific tasks, leveraging various libraries tailored to data retrieval, strategy implementation, performance evaluation, risk analysis, and greenwashing detection, enabling comprehensive analysis and informed decision-making.

### Conclusion
In conclusion, this thesis provides an overview of the efficient frontier and its importance in portfolio optimization. We also demonstrate how ESG data can be incorporated into the construction of the efficient frontier. We hope that this thesis and the accompanying code will serve as a useful resource for those interested in portfolio optimization and ESG investing.

### Requirements
To run the code the user needs to install the required packages found in the requirements.txt and a notebook environment, such as jupyter notebook, Visual studio code or use google colab.
Furhtermore the use needs to have installed the python language onto their machine. Python can be downloaded from: https://www.python.org/downloads/
To install the packages download the repository onto your machine.
From here either run the batch file called install_requirements.bat
or open a terminal and navigate with the terminal to the repository folder and run the command pip install -r requirements.txt
