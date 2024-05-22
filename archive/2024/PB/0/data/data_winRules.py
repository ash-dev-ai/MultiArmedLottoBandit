from fractions import Fraction

winning_data = {
    0: ('no win', Fraction(29220133879, 29220133800), 0), 
    9: ('r', Fraction(1, 3832), 4), 
    8: ('1w+r', Fraction(1, 9198), 4), 
    7: ('2w+r', Fraction(1, 70133), 7),  
    6: ('3w', Fraction(1, 57976), 7),  
    5: ('3w+r', Fraction(1, 1449411), 100),  
    4: ('4w', Fraction(1, 3652517), 100),  
    3: ('4w+r', Fraction(1, 91312918), 50000), 
    2: ('5w', Fraction(1, 1168805352), 1000000),  
    1: ('5w+r', Fraction(1, 29220133800), 20000000), 
}

# Accessing the data for a specific win type
win_type = 1
description, probability, usd_prize = winning_data[win_type]
print(f"Win Type {win_type}: {description}, Probability: {probability}, USD Prize: ${usd_prize}")
