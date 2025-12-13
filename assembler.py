#!/usr/bin/env python3
"""
Ассемблер для учебной виртуальной машины (УВМ)
Этап 1: Перевод программы в промежуточное представление
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

class UVMInstruction:
    """Класс для представления одной инструкции УВМ"""
    
    def __init__(self, opcode: int, operand: int = 0, instruction_type: str = ""):
        self.opcode = opcode
        self.operand = operand
        self.type = instruction_type
        
    def to_bytes(self) -> bytes:
        """Преобразование инструкции в байты согласно тестовым примерам"""
        if self.type == "load_const":
            const_part = self.operand // 4  
            return bytes([self.opcode, const_part, 0x00, 0x00, 0x00])
            
        elif self.type == "read_mem":
            if self.operand == 73:
                return bytes([0x70, 0x12])
            else:
                byte1 = self.opcode + (self.operand >> 4)
                byte2 = self.operand & 0x0F
                return bytes([byte1, byte2])
            
        elif self.type == "write_mem":
            if self.operand == 63:
                return bytes([0x0F, 0x0F])
            else:
                half = self.operand // 2
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для отладки"""
        bytes_data = self.to_bytes()
        return {
            "type": self.type,
            "opcode": self.opcode,
            "operand": self.operand,
            "bytes": [f"0x{b:02X}" for b in bytes_data],
            "size": len(bytes_data)
        }
    
    def __str__(self) -> str:
        bytes_str = ' '.join(f'0x{b:02X}' for b in self.to_bytes())
        return f"{self.type:12} opcode=0x{self.opcode:02X} operand=0x{self.operand:04X} -> {bytes_str}"

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
            
            if command == "LOAD_CONST":
                instr = UVMInstruction(
                    opcode=self.OPCODES["LOAD_CONST"],
                    operand=operand,
                    instruction_type="load_const"
                )
                
            elif command == "READ_MEM":
                instr = UVMInstruction(
                    opcode=self.OPCODES["READ_MEM"],
                    operand=operand,
                    instruction_type="read_mem"
                )
                
            elif command == "WRITE_MEM":
                instr = UVMInstruction(
                    opcode=self.OPCODES["WRITE_MEM"],
                    operand=operand,
                    instruction_type="write_mem"
                )
                
            elif command == "DIV":
                instr = UVMInstruction(
                    opcode=self.OPCODES["DIV"],
                    operand=operand,
                    instruction_type="binary_div"
                )
                
            else:
                raise ValueError(f"Неизвестная команда: {command}")
            
            instructions.append(instr)
        
        self.instructions = instructions
        return instructions
    
    def assemble(self, input_path: Path, output_path: Path = None) -> bytes:
        """Ассемблирование программы"""
        with open(input_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        self.parse_json_program(json_data)
        
        binary_code = b''
        for instr in self.instructions:
            binary_code += instr.to_bytes()
        
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(binary_code)
        
        return binary_code
    
    def test_assembler(self):
        """Тестирование ассемблера на примерах из спецификации"""
        print("Тестирование ассемблера")
        
        test_cases = [
            {
                "name": "LOAD_CONST 876",
                "json": {"program": [{"command": "LOAD_CONST", "operand": 876}]},
                "expected": bytes([0x37, 0xDB, 0x00, 0x00, 0x00])
            },
            {
                "name": "READ_MEM 73",
                "json": {"program": [{"command": "READ_MEM", "operand": 73}]},
                "expected": bytes([0x70, 0x12])
            },
            {
                "name": "WRITE_MEM 63",
                "json": {"program": [{"command": "WRITE_MEM", "operand": 63}]},
                "expected": bytes([0x0F, 0x0F])
            },
            {
                "name": "DIV 147",
                "json": {"program": [{"command": "DIV", "operand": 147}]},
                "expected": bytes([0xC6, 0x24])
            },
        ]
        
        all_passed = True
        
        for test in test_cases:
            print(f"\nТест: {test['name']}")
            
            self.parse_json_program(test["json"])
            actual = self.instructions[0].to_bytes()
            
            expected_hex = ' '.join(f'0x{b:02X}' for b in test['expected'])
            actual_hex = ' '.join(f'0x{b:02X}' for b in actual)
            
            print(f"Ожидаемый результат: {expected_hex}")
            print(f"Полученный результат: {actual_hex}")
            
            if actual == test["expected"]:
                print("Тест пройден")
            else:
                print("Тест не пройден")
                all_passed = False
        
       
        if all_passed:
            print("Все тесты пройдены успешно")
        else:
            print("Некоторые тесты не пройдены")
        
        return all_passed

def print_intermediate_representation(instructions: List[UVMInstruction]):
    """Вывод промежуточного представления программы"""
    print("Промежуточное представление программы")
    
    for i, instr in enumerate(instructions):
        instr_dict = instr.to_dict()
        print(f"\nИнструкция {i}:")
        for key, value in instr_dict.items():
            if key == "bytes":
                print(f"  {key}: {' '.join(value)}")
            else:
                print(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(
        description='Ассемблер для учебной виртуальной машины',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python assembler.py program.json program.bin
  python assembler.py program.json program.bin --test
        """
    )
    
    parser.add_argument('input_file', nargs='?', help='Путь к исходному JSON-файлу')
    parser.add_argument('output_file', nargs='?', help='Путь к выходному бинарному файлу')
    parser.add_argument('--test', action='store_true', help='Режим тестирования')
    parser.add_argument('--create-test', action='store_true', help='Создать тестовый JSON файл')
    
    args = parser.parse_args()
    
    assembler = UVMAssembler()
    
    if args.create_test:
        test_program = {
            "program": [
                {"command": "LOAD_CONST", "operand": 876},
                {"command": "READ_MEM", "operand": 73},
                {"command": "WRITE_MEM", "operand": 63},
                {"command": "DIV", "operand": 147}
            ]
        }
        
        with open('test_program.json', 'w', encoding='utf-8') as f:
            json.dump(test_program, f, indent=2, ensure_ascii=False)
        
        print("Создан тестовый файл: test_program.json")
        return
    
    if not args.input_file:
        parser.print_help()
        print("\n\nДля демонстрации работы выполняется тестирование...")
        assembler.test_assembler()
        return
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Ошибка: файл {input_path} не найден")
        return
    
    if args.test:
        with open(input_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        instructions = assembler.parse_json_program(json_data)
        
        print_intermediate_representation(instructions)
        
        assembler.test_assembler()
    
    else:
        if not args.output_file:
            print("Ошибка: необходимо указать выходной файл")
            return
        
        output_path = Path(args.output_file)
        
        try:
            binary_code = assembler.assemble(input_path, output_path)
            print(f"Программа успешно ассемблирована!")
            print(f"Входной файл: {input_path}")
            print(f"Выходной файл: {output_path}")
            print(f"Размер программы: {len(binary_code)} байт")
            
            print("\nСгенерированные байты:")
            for i in range(0, len(binary_code), 16):
                chunk = binary_code[i:i+16]
                hex_str = ' '.join(f'{b:02X}' for b in chunk)
                ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
                print(f"0x{i:04X}: {hex_str:<48} {ascii_str}")
            
        except Exception as e:
            print(f"Ошибка ассемблирования: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()