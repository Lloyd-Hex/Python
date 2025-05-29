import math #Импортируем библиотеку math

class Vector:# Инцилизация  класса Vector
    def __init__(self, coordinates):
        self.coordinates = coordinates  # Сохраняем координаты вектора в виде списка

    def length(self):
        return math.sqrt(sum(x**2 for x in self.coordinates))# Возвращает длину вектора

    def dot(self, other):
        return sum(a * b for a, b in zip(self.coordinates, other.coordinates))# Скалярное произведение двух векторов

    def angle_with(self, other):
        dot_product = self.dot(other)# Угол между двумя векторами в градусах
        len_self = self.length()
        len_other = other.length()
        if len_self == 0 or len_other == 0:
            raise ValueError("Один из векторов нулевой длины")
        cos_theta = dot_product / (len_self * len_other)
        return math.degrees(math.acos(cos_theta))

    def add(self, other):
        return Vector([a + b for a, b in zip(self.coordinates, other.coordinates)]) # Сложение двух векторов

    def subtract(self, other):
        return Vector([a - b for a, b in zip(self.coordinates, other.coordinates)]) # Вычитание двух векторов

    def multiply_by_scalar(self, scalar):
        return Vector([x * scalar for x in self.coordinates])# Умножение вектора на число

    def is_collinear(self, other):
        ratios = []# Проверка коллинеарности векторов (векторы пропорциональны)
        for a, b in zip(self.coordinates, other.coordinates):
            if b == 0:
                if a != 0:
                    return False
            else:
                ratios.append(a / b)
        return all(r == ratios[0] for r in ratios if b != 0)

class Matrix: # Создание класса Matrix
    def __init__(self, data):
        self.data = data  # Сохраняем матрицу как список
        self.rows = len(data)
        self.cols = len(data[0])

    def add(self, other):
        if self.rows != other.rows or self.cols != other.cols:# Сложение матриц
            raise ValueError("Размеры матриц не совпадают")
        return Matrix([[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def subtract(self, other):
        if self.rows != other.rows or self.cols != other.cols:# Вычитание матриц
            raise ValueError("Размеры матриц не совпадают")
        return Matrix([[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def multiply_by_vector(self, vector):
        if self.cols != len(vector.coordinates):# Умножение матрицы на вектор
            raise ValueError("Размеры не согласованы")
        result = []
        for row in self.data:
            result.append(sum(row[i] * vector.coordinates[i] for i in range(self.cols)))
        return Vector(result)

    def multiply(self, other):
        if self.cols != other.rows:# Умножение матриц
            raise ValueError("Число столбцов первой матрицы должно равняться числу строк второй")
        result = []
        for i in range(self.rows):
            row = []
            for j in range(other.cols):
                row.append(sum(self.data[i][k] * other.data[k][j] for k in range(self.cols)))
            result.append(row)
        return Matrix(result)

    def transpose(self):
        return Matrix([[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)])  # перестановка строк и столбцов

    def determinant(self):
        if self.rows != self.cols: # Нахождение определителя для квадратных матриц
            raise ValueError("Матрица должна быть квадратной")
        if self.rows == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        raise NotImplementedError("Поддерживаются только 2x2 матрицы")

    def gaussian_elimination(self):
        if self.rows != self.cols:  # поддержка для квадратных систем систем
            raise ValueError("Матрица должна быть квадратной")
        if self.rows != 2:
            raise NotImplementedError("Реализовано только для 2x2 матриц")
        a, b, c, d = self.data[0][0], self.data[0][1], self.data[1][0], self.data[1][1]
        e, f = self.data[0][2], self.data[1][2]
        det = a * d - b * c
        if det == 0:
            raise ValueError("Система не имеет единственного решения")
        x = (e * d - b * f) / det
        y = (a * f - e * c) / det
        return x, y


class EquationSolver:# Создание класса EquationSolver для уравнений
    def solve_linear(self, a, b):# ax + b = 0
        if a == 0:
            raise ValueError("a не может быть 0")
        return -b / a

    def solve_quadratic(self, a, b, c): # ax^2 + bx + c = 0
        if a == 0:
            return self.solve_linear(b, c)
        d = b ** 2 - 4 * a * c
        if d < 0:
            return "Нет вещественных корней"
        elif d == 0:
            x = -b / (2 * a)
            return x,
        else:
            x1 = (-b + math.sqrt(d)) / (2 * a)
            x2 = (-b - math.sqrt(d)) / (2 * a)
            return x1, x2

    def solve_system(self, matrix): # matrix объект представляющий расширенную матрицу системы
        return matrix.gaussian_elimination()


def main():# Интерфейс для пользователя
    solver = EquationSolver()

    while True:
        print("Сделайте выбор:")
        print("1. Работа с векторами")
        print("2. Работа с матрицами")
        print("3. Решение уравнений")
        print("0. Выход")
        choice = input("Сделайте выбор: ")

        if choice == "1":
            v1 = list(map(float, input("Координаты 1 вектора (через пробел): ").split()))
            v2 = list(map(float, input("Координаты 2 вектора (через пробел): ").split()))
            vec1 = Vector(v1)
            vec2 = Vector(v2)
            print("Сделайте выбор:")
            print("1. Длина вектора")
            print("2. Скалярное произведение")
            print("3. Угол между векторами")
            print("4. Сложение векторов")
            print("5. Вычитание векторов")
            print("6. Умножение на скаляр")
            print("7. Проверка коллинеарности")
            op = input("Операция: ")

            try:
                if op == "1":
                    print("Длина первого вектора:", vec1.length())
                    print("Длина второго вектора:", vec2.length())
                elif op == "2":
                    print("Скалярное произведение:", vec1.dot(vec2))
                elif op == "3":
                    print("Угол между векторами:", vec1.angle_with(vec2))
                elif op == "4":
                    print("Сумма:", vec1.add(vec2).coordinates)
                elif op == "5":
                    print("Разность:", vec1.subtract(vec2).coordinates)
                elif op == "6":
                    k = float(input("Введите скаляр: "))
                    print("Умножение первого:", vec1.multiply_by_scalar(k).coordinates)
                    print("Умножение второго:", vec2.multiply_by_scalar(k).coordinates)
                elif op == "7":
                    print("Коллинеарны:" if vec1.is_collinear(vec2) else "Не коллинеарны")
            except Exception as e:
                print("Ошибка:", e)

        elif choice == "2":
            rows, cols = map(int, input("Введите данные для матрицы (строки столбцы через пробел): ").split())
            print("Введите первую матрицу:")
            m1 = [list(map(float, input().split())) for _ in range(rows)]
            print("Введите вторую матрицу:")
            m2 = [list(map(float, input().split())) for _ in range(rows)]
            mat1 = Matrix(m1)
            mat2 = Matrix(m2)
            print("Сделайте выбор:")
            print("1. Сложение")
            print("2. Вычитание")
            print("3. Умножение")
            print("4. Транспонирование первой матрицы")
            op = input("Операция: ")

            try:
                if op == "1":
                    res = mat1.add(mat2)
                elif op == "2":
                    res = mat1.subtract(mat2)
                elif op == "3":
                    res = mat1.multiply(mat2)
                elif op == "4":
                    res = mat1.transpose()
                else:
                    continue
                for row in res.data:
                    print(*row)
            except Exception as e:
                print("Ошибка:", e)

        elif choice == "3":
            print("Тип уравнения:")
            print("1. Линейное (ax + b = 0)")
            print("2. Квадратное (ax² + bx + c = 0)")
            print("3. Система линейных уравнений (2x2)")
            op = input("Операция: ")
            try:
                if op == "1":
                    a, b = map(float, input("Введите a и b: ").split())
                    print("Решение:", solver.solve_linear(a, b))
                elif op == "2":
                    a, b, c = map(float, input("Введите a, b и c: ").split())
                    result = solver.solve_quadratic(a, b, c)
                    print("Решение:", result)
                elif op == "3":
                    print("Введите расширенную матрицу 2x3:")
                    m = [list(map(float, input().split())) for _ in range(2)]
                    matrix = Matrix(m)
                    x, y = solver.solve_system(matrix)
                    print(f"x = {x}, y = {y}")
            except Exception as e:
                print("Ошибка:", e)

        elif choice == "0":
            break

        else:
            print("Неверный выбор. Повторите.")


if __name__ == "__main__":# Интерфейс для пользователя
    main()

