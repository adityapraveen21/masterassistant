print ("I am calcbot, a calculator agent. i can help you solve simple mathematical expressions!")
x = True
while x:
    exp = input("Enter your mathematical expression of choice: ") 
    try:
        answer = eval(exp)
        print ("Your Answer is: ", answer)
    except:
        print("Invalid Mathematical Expression!")
    ch = input("Do you have another expression to be solved? (Y/N)")
    if ch.lower() == 'n':
        x = False
        print("Bye!")
    elif ch.lower() == 'y':
        continue
    else:
        print ("Invalid Choice!")
        x = False