def password_check(password):
    spec_symb = '!@#$%&*'
    if len(password) >= 8 and sum(char in spec_symb for char in password) >= 2:
        return 'Strong'
    else:
        return 'Weak'
password = input()
print(password_check(password))