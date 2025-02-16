def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

number = 9997
if is_prime(number):
    print(str(number) + " is prime.")
else:
    print(str(number) + " is not prime.")