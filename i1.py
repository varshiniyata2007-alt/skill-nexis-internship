{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "63854562-5b5c-42f4-b1a0-11cf99354f2a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset Information:\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 150 entries, 0 to 149\n",
      "Data columns (total 5 columns):\n",
      " #   Column        Non-Null Count  Dtype  \n",
      "---  ------        --------------  -----  \n",
      " 0   sepal_length  150 non-null    float64\n",
      " 1   sepal_width   150 non-null    float64\n",
      " 2   petal_length  150 non-null    float64\n",
      " 3   petal_width   150 non-null    float64\n",
      " 4   species       150 non-null    object \n",
      "dtypes: float64(4), object(1)\n",
      "memory usage: 6.0+ KB\n",
      "\n",
      "Basic Statistics:\n",
      "       sepal_length  sepal_width  petal_length  petal_width\n",
      "count    150.000000   150.000000    150.000000   150.000000\n",
      "mean       5.843333     3.057333      3.758000     1.199333\n",
      "std        0.828066     0.435866      1.765298     0.762238\n",
      "min        4.300000     2.000000      1.000000     0.100000\n",
      "25%        5.100000     2.800000      1.600000     0.300000\n",
      "50%        5.800000     3.000000      4.350000     1.300000\n",
      "75%        6.400000     3.300000      5.100000     1.800000\n",
      "max        7.900000     4.400000      6.900000     2.500000\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "url = \"https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv\"\n",
    "\n",
    "df = pd.read_csv(url)\n",
    "\n",
    "print(\"Dataset Information:\")\n",
    "df.info()\n",
    "\n",
    "print(\"\\nBasic Statistics:\")\n",
    "print(df.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "bcd4cd02-5aba-42eb-80b6-20b5e75e78ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Before imputation:\n",
      "    Age  Marks\n",
      "0  20.0   80.0\n",
      "1  21.0    NaN\n",
      "2   NaN   75.0\n",
      "3  22.0   90.0\n",
      "4  20.0   85.0\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "df = pd.DataFrame({\n",
    "    \"Age\": [20, 21, np.nan, 22, 20],\n",
    "    \"Marks\": [80, np.nan, 75, 90, 85]\n",
    "})\n",
    "\n",
    "print(\"Before imputation:\")\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "92419532-4e50-40ae-9f32-8367e97bc97b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Age      1\n",
      "Marks    1\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "f49f8e26-b1fc-4578-a295-2e4f31b8d720",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"Age\"] = df[\"Age\"].fillna(df[\"Age\"].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "0978cc93-815b-4a27-b422-3461112468f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"Marks\"] = df[\"Marks\"].fillna(df[\"Marks\"].median())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "9e721cb7-91f5-443e-a364-ead901d60d00",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "After imputation:\n",
      "     Age  Marks\n",
      "0  20.00   80.0\n",
      "1  21.00   82.5\n",
      "2  20.75   75.0\n",
      "3  22.00   90.0\n",
      "4  20.00   85.0\n"
     ]
    }
   ],
   "source": [
    "print(\"\\nAfter imputation:\")\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "903ba762-1c39-4887-92db-9101863b83af",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Before Encoding:\n",
      "   Gender  Age\n",
      "0    Male   20\n",
      "1  Female   21\n",
      "2  Female   22\n",
      "3    Male   20\n",
      "4  Female   23\n",
      "\n",
      "After Label Encoding:\n",
      "   Gender  Age\n",
      "0       1   20\n",
      "1       0   21\n",
      "2       0   22\n",
      "3       1   20\n",
      "4       0   23\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "# Sample dataset\n",
    "df = pd.DataFrame({\n",
    "    \"Gender\": [\"Male\", \"Female\", \"Female\", \"Male\", \"Female\"],\n",
    "    \"Age\": [20, 21, 22, 20, 23]\n",
    "})\n",
    "\n",
    "print(\"Before Encoding:\")\n",
    "print(df)\n",
    "\n",
    "# Label Encoding\n",
    "le = LabelEncoder()\n",
    "df[\"Gender\"] = le.fit_transform(df[\"Gender\"])\n",
    "\n",
    "print(\"\\nAfter Label Encoding:\")\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "4b03d13b-e06c-45e4-8956-0c2bd5954295",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Gender_Female  Gender_Male\n",
      "0            0.0          1.0\n",
      "1            1.0          0.0\n",
      "2            1.0          0.0\n",
      "3            0.0          1.0\n",
      "4            1.0          0.0\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "\n",
    "df = pd.DataFrame({\n",
    "    \"Gender\": [\"Male\", \"Female\", \"Female\", \"Male\", \"Female\"]\n",
    "})\n",
    "\n",
    "encoder = OneHotEncoder(sparse_output=False)\n",
    "\n",
    "encoded = encoder.fit_transform(df[[\"Gender\"]])\n",
    "\n",
    "encoded_df = pd.DataFrame(\n",
    "    encoded,\n",
    "    columns=encoder.get_feature_names_out([\"Gender\"])\n",
    ")\n",
    "\n",
    "print(encoded_df)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
