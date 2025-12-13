#!/usr/bin/env python3
"""
Интерпретатор для учебной виртуальной машины (УВМ) - Этап 3
Полный исправленный код с рабочими тестами
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class UVMInstruction:
    """Класс для представления одной инструкции УВМ"""
    type: str
    opcode: int
    operand: int
    address: int = 0
    
    def encode_safe(self) -> bytes:
        """Безопасное кодирование с проверкой значений"""
        if self.type == "load_const":
            operand_low = self.operand & 0xFF
            return bytes([self.opcode, operand_low, 0x00, 0x00, 0x00])
            
        elif self.type == "read_mem":
            operand_byte = self.operand & 0xFF
            return bytes([0x30, operand_byte])  
            
        elif self.type == "write_mem":
            operand_byte = self.operand & 0xFF
            return bytes([0x1F, operand_byte])  
            
        elif self.type == "binary_div":
            operand_byte = self.operand & 0xFF
            return bytes([0x06, operand_byte])  
            
        else:
            raise ValueError(f"Неизвестный тип инструкции: {self.type}")
    
    def get_size(self) -> int:
        """Получить размер команды в байтах"""
        return 5 if self.type == "load_const" else 2
    
    def __str__(self) -> str:
        return f"{self.type}(opcode=0x{self.opcode:02X}, operand=0x{self.operand:X})"


@dataclass
class UVMMemory:
    """Модель памяти УВМ"""
    code_memory: bytearray = field(default_factory=lambda: bytearray(65536))
    data_memory: bytearray = field(default_factory=lambda: bytearray(65536))
    stack: List[int] = field(default_factory=list)
    
    def load_code(self, code: bytes, offset: int = 0):
        """Загрузка кода в память команд"""
        end = offset + len(code)
        if end > len(self.code_memory):
            self.code_memory.extend(bytearray(end - len(self.code_memory)))
        self.code_memory[offset:end] = code
    
    def read_data(self, address: int, size: int = 4) -> int:
        """Чтение данных из памяти"""
        if address < 0:
            return 0
        
        if address + size > len(self.data_memory):
            self.data_memory.extend(bytearray(address + size - len(self.data_memory)))
       
        value = 0
        for i in range(size):
            value = (value << 8) | self.data_memory[address + i]
        return value
    
    def write_data(self, address: int, value: int, size: int = 4):
        """Запись данных в память"""
        if address < 0:
            return
        
        if address + size > len(self.data_memory):
            self.data_memory.extend(bytearray(address + size - len(self.data_memory)))
        
        for i in range(size - 1, -1, -1):
            self.data_memory[address + i] = value & 0xFF
            value >>= 8
    
    def push(self, value: int):
        """Поместить значение на стек"""
        self.stack.append(value)
    
    def pop(self) -> int:
        """Снять значение с вершины стека"""
        if not self.stack:
            return 0
        return self.stack.pop()
    
    def peek(self) -> int:
        """Посмотреть значение на вершине стека без извлечения"""
        if not self.stack:
            return 0
        return self.stack[-1]
    
    def clear_stack(self):
        """Очистить стек"""
        self.stack.clear()
    
    def get_memory_dump(self, start_addr: int, end_addr: int) -> Dict[str, Any]:
        """Получить дамп памяти в формате JSON"""
        if start_addr < 0 or start_addr > end_addr:
            start_addr = 0
            end_addr = min(255, len(self.data_memory) - 1)
        
        if end_addr >= len(self.data_memory):
            self.data_memory.extend(bytearray(end_addr - len(self.data_memory) + 1))
        
        dump = {
            "metadata": {
                "range": f"0x{start_addr:04X}-0x{end_addr:04X}",
                "total_bytes": end_addr - start_addr + 1,
                "stack_size": len(self.stack),
                "stack_values": self.stack.copy()
            },
            "memory": []
        }
        
        for addr in range(start_addr, end_addr + 1):
            dump["memory"].append({
                "address": f"0x{addr:04X}",
                "decimal": self.data_memory[addr],
                "hex": f"0x{self.data_memory[addr]:02X}",
                "ascii": chr(self.data_memory[addr]) if 32 <= self.data_memory[addr] < 127 else "."
            })
        
        return dump
    
    def save_memory_dump(self, filepath: Path, start_addr: int, end_addr: int):
        """Сохранить дамп памяти в JSON файл"""
        dump = self.get_memory_dump(start_addr, end_addr)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dump, f, indent=2, ensure_ascii=False)


class UVMInterpreter:
    """Интерпретатор УВМ"""
    
    def __init__(self):
        self.memory = UVMMemory()
        self.pc = 0
        self.running = False
        self.instructions_executed = 0
        self.verbose = True
    
    def set_verbose(self, verbose: bool):
        """Установить режим подробного вывода"""
        self.verbose = verbose
    
    def load_program(self, binary_file: Path):
        """Загрузка программы из бинарного файла"""
        with open(binary_file, 'rb') as f:
            code = f.read()
        
        self.memory.load_code(code, 0)
        self.pc = 0
        self.instructions_executed = 0
        self.memory.clear_stack()
        if self.verbose:
            print(f"Загружена программа из {binary_file}")
            print(f"  Размер: {len(code)} байт")
    
    def execute_simple_program(self, instructions: List[UVMInstruction]) -> bool:
        """Выполнение простой программы из списка инструкций"""
        if self.verbose:
            print(f"\nВыполнение программы")
        
        self.instructions_executed = 0
        
        for instr in instructions:
            self.instructions_executed += 1
            
            if instr.type == "load_const":
                self.memory.push(instr.operand)
                if self.verbose:
                    print(f"[{self.instructions_executed:3d}] LOAD_CONST {instr.operand} -> стек[{len(self.memory.stack)-1}]")
                
            elif instr.type == "read_mem":
                if len(self.memory.stack) < 1:
                    print(f"Ошибка: стек пуст для READ_MEM")
                    return False
                
                base_addr = self.memory.pop()
                offset = instr.operand
                address = base_addr + offset
                value = self.memory.read_data(address, 4)
                self.memory.push(value)
                
                if self.verbose:
                    print(f"[{self.instructions_executed:3d}] READ_MEM [0x{base_addr:X}+{offset}] = [0x{address:X}] -> {value} на стек")
                
            elif instr.type == "write_mem":
                if len(self.memory.stack) < 1:
                    print(f"Ошибка: стек пуст для WRITE_MEM")
                    return False
                
                value = self.memory.pop()
                address = instr.operand
                self.memory.write_data(address, value, 4)
                
                if self.verbose:
                    print(f"[{self.instructions_executed:3d}] WRITE_MEM {value} -> [0x{address:X}]")
                
            elif instr.type == "binary_div":
                if len(self.memory.stack) < 2:
                    print(f"Ошибка: недостаточно значений на стеке для DIV")
                    return False
                
                b = self.memory.pop()
                a = self.memory.pop()
                if b == 0:
                    print(f"Ошибка: деление на ноль")
                    return False
                
                result = a // b
                address = instr.operand
                self.memory.write_data(address, result, 4)
                self.memory.push(result)
                
                if self.verbose:
                    print(f"[{self.instructions_executed:3d}] DIV {a}/{b} = {result} -> [0x{address:X}] и на стек")
            
            else:
                print(f"Неизвестная инструкция: {instr.type}")
                return False
        
        if self.verbose:
            print(f"\nПрограмма выполнена успешно!")
            print(f"  Выполнено инструкций: {self.instructions_executed}")
            print(f"  Размер стека: {len(self.memory.stack)}")
            
            if self.memory.stack:
                print(f"  Состояние стека: {self.memory.stack}")
        
        return True


def run_simple_copy_test() -> bool:
    """Простой тест копирования одного значения"""
    print(f"\nПростой тест копирования")
    
    interpreter = UVMInterpreter()
    interpreter.set_verbose(True)
    
    source_addr = 0x1000
    dest_addr = 0x2000
    test_value = 12345
    
    print(f"Тестовое значение: {test_value}")
    print(f"Адрес источника: 0x{source_addr:04X}")
    print(f"Адрес назначения: 0x{dest_addr:04X}")
    
    interpreter.memory.write_data(source_addr, test_value, 4)
    print(f"Записано {test_value} в источник (0x{source_addr:04X})")
    
    program = [
        UVMInstruction("load_const", 55, source_addr),
        UVMInstruction("read_mem", 48, 0),
        UVMInstruction("write_mem", 31, dest_addr),
    ]
    
    print(f"\nВыполнение программы копирования...")
    success = interpreter.execute_simple_program(program)
    
    if not success:
        print(f"\nОшибка выполнения программы")
        return False
    
    print(f"\n")
    print(f"Проверка результата:")
    
    source_val = interpreter.memory.read_data(source_addr, 4)
    dest_val = interpreter.memory.read_data(dest_addr, 4)
    
    print(f"  Значение в источнике (0x{source_addr:04X}): {source_val}")
    print(f"  Значение в назначении (0x{dest_addr:04X}): {dest_val}")
    
    if source_val == dest_val and source_val == test_value:
        print(f"\nТест пройден: Значение успешно скопировано!")
        return True
    else:
        print(f"\nТест не пройден:")
        print(f"  Ожидалось: {test_value}")
        print(f"  В источнике: {source_val}")
        print(f"  В назначении: {dest_val}")
        return False


def run_array_copy_test() -> bool:
    """Тест копирования массива из 5 элементов"""
    print(f"\nТест копирования массива")
    
    interpreter = UVMInterpreter()
    interpreter.set_verbose(True)
    
    source_base = 0x1000
    dest_base = 0x2000
    array_data = [10, 20, 30, 40, 50]
    
    print(f"Исходный массив (5 элементов): {array_data}")
    print(f"Адрес источника: 0x{source_base:04X}")
    print(f"Адрес назначения: 0x{dest_base:04X}")
    
    print(f"\nИнициализация памяти:")
    for i, value in enumerate(array_data):
        addr = source_base + i * 4
        interpreter.memory.write_data(addr, value, 4)
        print(f"  Записано {value} по адресу 0x{addr:04X}")
    
    print(f"\nСоздание программы копирования...")
    program = []
    
    for i in range(len(array_data)):
        src_addr = source_base + i * 4
        dst_addr = dest_base + i * 4
        
        program.append(UVMInstruction("load_const", 55, src_addr))  
        program.append(UVMInstruction("read_mem", 48, 0))           
        program.append(UVMInstruction("write_mem", 31, dst_addr))   
    
    print(f"\nВыполнение программы...")
    success = interpreter.execute_simple_program(program)
    
    if not success:
        print(f"\nОшибка выполнения программы")
        return False
    
    print(f"\n")
    print(f"Проверка результата:")
    
    all_correct = True
    
    print(f"\nИсходный массив:")
    source_values = []
    for i in range(len(array_data)):
        addr = source_base + i * 4
        value = interpreter.memory.read_data(addr, 4)
        source_values.append(value)
        print(f"  0x{addr:04X}: {value}")
    
    print(f"\nСкопированный массив:")
    copied_data = []
    for i in range(len(array_data)):
        addr = dest_base + i * 4
        value = interpreter.memory.read_data(addr, 4)
        copied_data.append(value)
        print(f"  0x{addr:04X}: {value}")
        
        if value != array_data[i]:
            all_correct = False
    
    if all_correct:
        print(f"\nТест пройден: Весь массив успешно скопирован!")
        return True
    else:
        print(f"\nТест не пройдет:")
        print(f"  Ожидалось: {array_data}")
        print(f"  Получено: {copied_data}")
        return False


def run_comprehensive_test() -> bool:
    """Комплексный тест всех операций"""
    print(f"\nКомплексный тест всех операций")
    
    interpreter = UVMInterpreter()
    interpreter.set_verbose(True)
    
    print(f"Тест включает:")
    print(f"  1. Загрузку констант")
    print(f"  2. Запись в память")
    print(f"  3. Чтение из памяти")
    print(f"  4. Операцию деления")
    
    program = [
        UVMInstruction("load_const", 55, 100),
        UVMInstruction("write_mem", 31, 0x1000),
        
        UVMInstruction("load_const", 55, 200),
        UVMInstruction("write_mem", 31, 0x1004),
        
        UVMInstruction("load_const", 55, 0x1000),
        UVMInstruction("read_mem", 48, 0),
        UVMInstruction("write_mem", 31, 0x2000),
        
        UVMInstruction("load_const", 55, 0x1004),
        UVMInstruction("read_mem", 48, 0),
        UVMInstruction("write_mem", 31, 0x2004),
        
        UVMInstruction("load_const", 55, 100),
        UVMInstruction("load_const", 55, 25),
        UVMInstruction("binary_div", 6, 0x3000),
    ]
    
    print(f"\nВыполнение комплексной программы...")
    success = interpreter.execute_simple_program(program)
    
    if not success:
        print(f"\nОшибка выполнения программы")
        return False
    
    print(f"\n")
    print(f"Проверка результатов")
    
    tests_passed = 0
    total_tests = 3
    
    print(f"\n1. Проверка записанных значений:")
    val1 = interpreter.memory.read_data(0x1000, 4)
    val2 = interpreter.memory.read_data(0x1004, 4)
    
    print(f"  0x1000: {val1} (ожидалось: 100)")
    print(f"  0x1004: {val2} (ожидалось: 200)")
    
    if val1 == 100 and val2 == 200:
        print(f"  Тест 1 пройден")
        tests_passed += 1
    else:
        print(f"  Тест 1 не пройден")
    
    print(f"\n2. Проверка копирования:")
    copy1 = interpreter.memory.read_data(0x2000, 4)
    copy2 = interpreter.memory.read_data(0x2004, 4)
    
    print(f"  0x2000: {copy1} (ожидалось: 100)")
    print(f"  0x2004: {copy2} (ожидалось: 200)")
    
    if copy1 == 100 and copy2 == 200:
        print(f"  Тест 2 пройден")
        tests_passed += 1
    else:
        print(f"  Тест 2 не пройден")
    
    print(f"\n3. Проверка деления:")
    div_result = interpreter.memory.read_data(0x3000, 4)
    
    print(f"  0x3000: {div_result} (ожидалось: 4, т.к. 100 ÷ 25 = 4)")
    
    if div_result == 4:
        print(f"  Тест 3 пройден")
        tests_passed += 1
    else:
        print(f"  Тест 3 не пройден")
    
    print(f"\n")
    print(f"ИТОГ: {tests_passed}/{total_tests} тестов пройдено")
    
    if tests_passed == total_tests:
        print(f"\nВсе тесты пройдены успешно")
        return True
    else:
        print(f"\nНекоторые тесты не пройдены")
        return False


def create_test_program():
    """Создание тестовой программы и сохранение дампа памяти"""
    print(f"\n{'='*70}")
    print(f"Создание тестовой программы и дампа памяти")
    print(f"{'='*70}")
    
    interpreter = UVMInterpreter()
    interpreter.set_verbose(False)  
    
    program = [
        UVMInstruction("load_const", 55, 0x1234),
        UVMInstruction("write_mem", 31, 0x1000),
        UVMInstruction("load_const", 55, 0x5678),
        UVMInstruction("write_mem", 31, 0x1004),
        UVMInstruction("load_const", 55, 0x1000),
        UVMInstruction("read_mem", 48, 0),
        UVMInstruction("write_mem", 31, 0x2000),
    ]
    
    print(f"Выполнение тестовой программы")
    success = interpreter.execute_simple_program(program)
    
    if not success:
        print(f"Ошибка выполнения программы")
        return False
    
    output_file = Path("memory_dump.json")
    interpreter.memory.save_memory_dump(output_file, 0x0F00, 0x1100)
    
    print(f"\nДамп памяти сохранен в файл: {output_file}")
    print(f"  Диапазон: 0x0F00 - 0x1100")
    
    print(f"\nПредпросмотр дампа (первые 10 записей):")
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    for i, item in enumerate(data["memory"][:10]):
        print(f"  {item['address']}: {item['hex']} ({item['decimal']})")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Интерпретатор для учебной виртуальной машины (УВМ) - Этап 3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Простой тест копирования
  python interpreter.py --simple
  
  # Тест копирования массива
  python interpreter.py --array
  
  # Комплексный тест всех операций
  python interpreter.py --comprehensive
  
  # Создание тестовой программы и дампа памяти
  python interpreter.py --create-dump
  
  # Запуск всех тестов
  python interpreter.py --all
  
  # Демонстрация работы
  python interpreter.py --demo
        """
    )
    
    parser.add_argument('--simple', action='store_true', help='Простой тест копирования')
    parser.add_argument('--array', action='store_true', help='Тест копирования массива')
    parser.add_argument('--comprehensive', action='store_true', help='Комплексный тест всех операций')
    parser.add_argument('--create-dump', action='store_true', help='Создать тестовую программу и дамп памяти')
    parser.add_argument('--all', action='store_true', help='Запустить все тесты')
    parser.add_argument('--demo', action='store_true', help='Демонстрация работы')
    
    args = parser.parse_args()
    
    if args.simple:
        success = run_simple_copy_test()
        sys.exit(0 if success else 1)
    
    elif args.array:
        success = run_array_copy_test()
        sys.exit(0 if success else 1)
    
    elif args.comprehensive:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    
    elif args.create_dump:
        success = create_test_program()
        sys.exit(0 if success else 1)
    
    elif args.all:
        print(f"Запуск всех тестов")
        
        results = []
        
        print(f"\n1. Простой тест копирования:")
        results.append(run_simple_copy_test())
        
        print(f"\n\n2. Тест копирования массива:")
        results.append(run_array_copy_test())
        
        print(f"\n\n3. Комплексный тест всех операций:")
        results.append(run_comprehensive_test())
        
        passed = sum(1 for r in results if r)
        total = len(results)
        
        print(f"Итог всех тестов: {passed}/{total} пройдено")
        
        if passed == total:
            print(f"Все тесты пройдены успешно")
            sys.exit(0)
        else:
            print(f"Некоторые тесты не пройдены")
            sys.exit(1)
    
    elif args.demo:
        print(f"Демонстрация работы интерпретатора УВМ")
        
        print(f"\n1. Простой тест копирования:")
        run_simple_copy_test()
        
        print(f"\n\n2. Тест копирования массива:")
        run_array_copy_test()
        
        print(f"\n\n3. Создание дампа памяти:")
        create_test_program()
        
        print(f"\n")
        print(f"Демонстрация завершена")
        
        return
    
    else:
        parser.print_help()
        print(f"\nЗапуск демонстрации")
        
        print(f"Демонстрация работы интерпретатора УВМ")
        
        print(f"\n1. Простой тест копирования:")
        run_simple_copy_test()
        
        print(f"\n\n2. Тест копирования массива:")
        run_array_copy_test()


if __name__ == "__main__":
    main()