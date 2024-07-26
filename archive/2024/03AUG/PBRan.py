# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 18:34:01 2023

@author: AVMal
"""

import random

def generate_numbers():
    w_numbers = []
    r_number = random.randint(1, 26)
    
    while len(w_numbers) < 5:
        number = random.randint(1, 69)
        if number not in w_numbers:
            w_numbers.append(number)
    
    return w_numbers, r_number

def generate_tickets(num_tickets):
    tickets = []
    for _ in range(num_tickets):
        w_numbers, r_number = generate_numbers()
        ticket = {"W Numbers": w_numbers, "R Number": r_number}
        tickets.append(ticket)
    
    return tickets

num_tickets = 10
tickets = generate_tickets(num_tickets)

# Printing the generated tickets
for idx, ticket in enumerate(tickets, 1):
    print(f"Ticket {idx}:")
    print("W Numbers:", ticket["W Numbers"])
    print("R Number:", ticket["R Number"])
    print()