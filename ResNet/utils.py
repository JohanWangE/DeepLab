import datetime

def write_file_and_close(filename, *arg, flag = "a"):
    with open(filename, flag) as output_file:
        output_file.write(str(datetime.datetime.now()))
        output_file.write(":\n")
        output_file.write(*arg)
        output_file.write("\n")
        print(*arg)

def check_control(filename):
    with open(filename, "r") as filename:
        try:
            u = int(filename.read().strip())
            return bool(u)
        except:
            write_file_and_close("Error occured checking control!!!")
            return False
