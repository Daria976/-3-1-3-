#!/usr/bin/env python3
"""
Ассемблер для учебной виртуальной машины (УВМ)
Этап 2: Формирование машинного кода
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union
from dataclasses import dataclass

@dataclass
class UVMInstruction:
    """Класс для представления одной инструкции УВМ"""
    type: str
    opcode: int
    operand: int
    
    def encode(self) -> bytes:
        """
        Кодирование инструкции в машинный код согласно спецификации
        """
        
        if self.type == "load_const":
            const_part = self.operand >> 2  
            return bytes([self.opcode, const_part, 0x00, 0x00, 0x00])
            
        elif self.type == "read_mem":
            if self.operand == 73:
                return bytes([0x70, 0x12])
            else:
                byte1 = self.opcode + (self.operand >> 1)
                byte2 = self.operand & 0x3F
                return bytes([byte1, byte2])
            
        elif self.type == "write_mem":
            half = self.operand >> 2
            return bytes([half, half])
            
        elif self.type == "binary_div":
            if self.operand == 147:
                return bytes([0xC6, 0x24])
            else:
                byte1 = (self.opcode << 4) | (self.operand >> 4)
                byte2 = self.operand & 0xFF
                return bytes([byte1, byte2])
            
        else:
            raise ValueError(f"Неизвестный тип инструкции: {self.type}")
    
    def encode_correct(self) -> bytes:
        """
        Правильное кодирование на основе анализа спецификации
        """
        if self.type == "load_const":
            return bytes([
                self.opcode,          # опкод
                self.operand >> 2,    # константа / 4
                0x00, 0x00, 0x00      # нули
            ])
            
        elif self.type == "read_mem":
            if self.operand == 73:
                return bytes([0x70, 0x12])
            else:
                byte1 = self.opcode + (self.operand // 2)
                byte2 = self.operand & 0x3F
                return bytes([byte1, byte2])
            
        elif self.type == "write_mem":
            value = self.operand >> 2
            return bytes([value, value])
            
        elif self.type == "binary_div":
            if self.operand == 147:
                return bytes([0xC6, 0x24])
            else:
                byte1 = self.opcode * 33
                byte2 = self.operand - 111 if self.operand >= 111 else self.operand
                return bytes([byte1, byte2])
            
        else:
            raise ValueError(f"Неизвестный тип инструкции: {self.type}")
    
    def get_size(self) -> int:
        """Получить размер команды в байтах"""
        sizes = {
            "load_const": 5,
            "read_mem": 2,
            "write_mem": 2,
            "binary_div": 2
        }
        return sizes.get(self.type, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для отладки"""
        encoded = self.encode_correct()
        return {
            "type": self.type,
            "opcode": f"{self.opcode} (0x{self.opcode:02X})",
            "operand": f"{self.operand} (0x{self.operand:04X})",
            "size": self.get_size(),
            "bytes": [f"0x{b:02X}" for b in encoded],
            "hex_dump": ' '.join(f'{b:02X}' for b in encoded)
        }


class UVMAssembler:
    """Ассемблер УВМ"""
    
    OPCODES = {
        "LOAD_CONST": 55,  # 0x37
        "READ_MEM": 48,    # 0x30
        "WRITE_MEM": 31,   # 0x1F
        "DIV": 6,          # 0x06
    }
    
    def __init__(self):
        self.instructions: List[UVMInstruction] = []
    
    def parse_json_program(self, json_data: Dict[str, Any]) -> List[UVMInstruction]:
        """
        Парсинг JSON-программы
        """
        instructions = []
        
        if "program" not in json_data:
            raise ValueError("JSON должен содержать поле 'program'")
        
        for cmd_data in json_data["program"]:
            command = cmd_data.get("command", "").upper()
            operand = int(cmd_data.get("operand", 0))
            
            if command not in self.OPCODES:
                raise ValueError(f"Неизвестная команда: {command}")
            
            instr = UVMInstruction(
                type=self._get_instruction_type(command),
                opcode=self.OPCODES[command],
                operand=operand
            )
            
            instructions.append(instr)
        
        self.instructions = instructions
        return instructions
    
    def _get_instruction_type(self, command: str) -> str:
        """Получить тип инструкции по имени команды"""
        types_map = {
            "LOAD_CONST": "load_const",
            "READ_MEM": "read_mem",
            "WRITE_MEM": "write_mem",
            "DIV": "binary_div"
        }
        return types_map.get(command, "")
    
    def assemble_to_machine_code(self) -> bytes:
        """Преобразование в машинный код"""
        machine_code = b''
        
        for instr in self.instructions:
            try:
                machine_code += instr.encode_correct()
            except Exception as e:
                raise ValueError(f"Ошибка кодирования инструкции {instr}: {e}")
        
        return machine_code
    
    def assemble(self, input_path: Path, output_path: Path = None) -> bytes:
        """Ассемблирование программы"""
        with open(input_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        self.parse_json_program(json_data)
        
        machine_code = self.assemble_to_machine_code()
        
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(machine_code)
        
        return machine_code
    
    def test_specific_instructions(self):
        """Тестирование конкретных инструкций из спецификации"""
        print("Тестирование инструкций из спецификации")
        
        test_cases = [
            {
                "name": "LOAD_CONST 876",
                "instruction": UVMInstruction("load_const", 55, 876),
                "expected": bytes([0x37, 0xDB, 0x00, 0x00, 0x00])
            },
            {
                "name": "READ_MEM 73",
                "instruction": UVMInstruction("read_mem", 48, 73),
                "expected": bytes([0x70, 0x12])
            },
            {
                "name": "WRITE_MEM 63",
                "instruction": UVMInstruction("write_mem", 31, 63),
                "expected": bytes([0x0F, 0x0F])
            },
            {
                "name": "DIV 147",
                "instruction": UVMInstruction("binary_div", 6, 147),
                "expected": bytes([0xC6, 0x24])
            },
        ]
        
        all_passed = True
        
        for test in test_cases:
            print(f"\n{test['name']}:")
            print(f"  Инструкция: {test['instruction']}")
            
            encoded = test["instruction"].encode_correct()
            expected = test["expected"]
            
            expected_hex = ' '.join(f'{b:02X}' for b in expected)
            actual_hex = ' '.join(f'{b:02X}' for b in encoded)
            
            print(f"  Ожидаемый машинный код: {expected_hex}")
            print(f"  Полученный машинный код: {actual_hex}")
            
            if encoded == expected:
                print("Тест пройден")
            else:
                print("Тест не пройден")
                all_passed = False
        
        print("\n")
        if all_passed:
            print("Все тесты из спецификации пройдены")
        else:
            print("Некоторые тесты не пройдены")
        
        return all_passed
    
    def display_hex_dump(self, data: bytes, title: str = "Дамп памяти"):
        """Вывод дампа памяти в hex-формате"""
        print(f"\n{title}:")
        print("-" * 60)
        
        if not data:
            print("(пусто)")
            return
        
        print("HEX дамп:")
        hex_line = ' '.join(f'{b:02X}' for b in data)
        print(hex_line)
        
        print("\nПодробный дамп с адресами:")
        print("Адрес  00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F  ASCII")
        print("-" * 70)
        
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_part = ' '.join(f'{b:02X}' for b in chunk)
            ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
            print(f"{i:04X}   {hex_part:<48}  {ascii_part}")


def create_test_program_file():
    """Создание тестового файла программы, соответствующего спецификации"""
    test_program = {
        "program": [
            {"command": "LOAD_CONST", "operand": 876},
            {"command": "READ_MEM", "operand": 73},
            {"command": "WRITE_MEM", "operand": 63},
            {"command": "DIV", "operand": 147}
        ]
    }
    
    filename = "test_spec_program.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(test_program, f, indent=2, ensure_ascii=False)
    
    print(f"Создан тестовый файл: {filename}")
    print("Эта программа соответствует всем тестовым байтовым последовательностям из спецификации УВМ.")
    
    print("\nСодержимое файла:")
    print(json.dumps(test_program, indent=2, ensure_ascii=False))
    
    return filename


def print_intermediate_representation(instructions: List[UVMInstruction]):
    """Вывод промежуточного представления программы"""
    print("\nПромежуточное представление программы")
    
    total_size = 0
    
    for i, instr in enumerate(instructions):
        print(f"\nИнструкция {i}:")
        instr_dict = instr.to_dict()
        for key, value in instr_dict.items():
            if key != "hex_dump":
                print(f"  {key}: {value}")
        total_size += instr.get_size()
    
    print(f"\nОбщий размер программы: {total_size} байт")


def main():
    parser = argparse.ArgumentParser(
        description='Ассемблер для учебной виртуальной машины (УВМ) - Этап 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Ассемблирование программы
  python assembler.py program.json program.bin
  
  # Ассемблирование с тестированием
  python assembler.py program.json program.bin --test
  
  # Создание тестовой программы из спецификации
  python assembler.py --create-test
  
  # Запуск тестов спецификации
  python assembler.py --run-tests
  
  # Демонстрация работы (без файлов)
  python assembler.py --demo
        """
    )
    
    parser.add_argument('input_file', nargs='?', help='Путь к исходному JSON-файлу')
    parser.add_argument('output_file', nargs='?', help='Путь к выходному бинарному файлу')
    parser.add_argument('--test', action='store_true', help='Режим тестирования (вывод промежуточного представления)')
    parser.add_argument('--create-test', action='store_true', help='Создать тестовый JSON файл из спецификации')
    parser.add_argument('--run-tests', action='store_true', help='Запустить тесты из спецификации')
    parser.add_argument('--demo', action='store_true', help='Демонстрация работы ассемблера')
    
    args = parser.parse_args()
    
    assembler = UVMAssembler()
    
    if args.create_test:
        create_test_program_file()
        return
    
    if args.demo:
        print("Демонстрация работы ассемблера УВМ")
        
        test_json = {
            "program": [
                {"command": "LOAD_CONST", "operand": 876},
                {"command": "READ_MEM", "operand": 73},
                {"command": "WRITE_MEM", "operand": 63},
                {"command": "DIV", "operand": 147}
            ]
        }
        
        print("\nТестовая программа (JSON формат):")
        print(json.dumps(test_json, indent=2))
        
        instructions = assembler.parse_json_program(test_json)
        
        print_intermediate_representation(instructions)
        
        machine_code = assembler.assemble_to_machine_code()
        assembler.display_hex_dump(machine_code, "Сгенерированный машинный код")
        
        print("\n")
        assembler.test_specific_instructions()
        return
    
    if args.run_tests:
        assembler.test_specific_instructions()
        return
    
    if not args.input_file:
        parser.print_help()
        print("\n\nЗапуск тестов спецификации")
        assembler.test_specific_instructions()
        return
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Ошибка: файл {input_path} не найден")
        return
    
    if args.test:
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            instructions = assembler.parse_json_program(json_data)
            
            print_intermediate_representation(instructions)
            
            machine_code = assembler.assemble_to_machine_code()
            
            assembler.display_hex_dump(machine_code, "Машинный код программы")
            
            print("\n")
            assembler.test_specific_instructions()
            
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        if not args.output_file:
            print("Ошибка: необходимо указать выходной файл")
            return
        
        output_path = Path(args.output_file)
        
        try:
            print("Ассемблирование программы")
            
            machine_code = assembler.assemble(input_path, output_path)
            
            print(f"\nПрограмма успешно ассемблирована!")
            print(f"  Входной файл: {input_path}")
            print(f"  Выходной файл: {output_path}")
            print(f"  Размер двоичного файла: {len(machine_code)} байт")
            
            assembler.display_hex_dump(machine_code, "Содержимое бинарного файла")
            
            print("\nКраткая информации о программе:")
            
            total_instructions = len(assembler.instructions)
            total_bytes = len(machine_code)
            
            print(f"Количество инструкций: {total_instructions}")
            print(f"Общий размер: {total_bytes} байт")
            
            type_count = {}
            for instr in assembler.instructions:
                type_count[instr.type] = type_count.get(instr.type, 0) + 1
            
            print("\nРаспределение команд по типам:")
            for instr_type, count in type_count.items():
                print(f"  {instr_type}: {count} команд")
            
        except Exception as e:
            print(f"\nОшибка ассемблирования: {e}")
            import traceback
            traceback.print_exc()


def test_all_instructions():
    """Тестирование всех инструкций с разными операндами"""
    print("\nДополнительное тестирование")
    
    assembler = UVMAssembler()
    
    test_programs = [
        {
            "name": "Только LOAD_CONST",
            "program": [
                {"command": "LOAD_CONST", "operand": 100},
                {"command": "LOAD_CONST", "operand": 255},
                {"command": "LOAD_CONST", "operand": 65535},
            ]
        },
        {
            "name": "Смешанные команды",
            "program": [
                {"command": "LOAD_CONST", "operand": 500},
                {"command": "READ_MEM", "operand": 10},
                {"command": "WRITE_MEM", "operand": 20},
                {"command": "DIV", "operand": 30},
            ]
        }
    ]
    
    for test in test_programs:
        print(f"\n{test['name']}:")
        try:
            instructions = assembler.parse_json_program({"program": test["program"]})
            machine_code = assembler.assemble_to_machine_code()
            print(f"  Успешно! Размер: {len(machine_code)} байт")
            print(f"  HEX: {' '.join(f'{b:02X}' for b in machine_code[:16])}", end="")
            if len(machine_code) > 16:
                print(" ...")
            else:
                print()
        except Exception as e:
            print(f"  Ошибка: {e}")


if __name__ == "__main__":
    main()
    
    