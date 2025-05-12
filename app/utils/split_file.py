import os
import argparse

def split_file(input_file, output_dir=None, lines_per_file=500):
    """
    Tách một file văn bản thành nhiều file nhỏ hơn, mỗi file chứa số dòng được chỉ định.
    
    Args:
        input_file (str): Đường dẫn đến file cần tách
        output_dir (str, optional): Thư mục đầu ra để lưu các file đã tách. 
                                   Nếu không được chỉ định, sẽ sử dụng thư mục của file đầu vào.
        lines_per_file (int, optional): Số dòng trong mỗi file đầu ra. Mặc định là 500.
    
    Returns:
        list: Danh sách các file đã được tạo
    """
    # Kiểm tra file đầu vào tồn tại
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Không tìm thấy file: {input_file}")
    
    # Xác định thư mục đầu ra
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy tên file gốc không có phần mở rộng
    base_name = os.path.basename(input_file)
    file_name, file_ext = os.path.splitext(base_name)
    
    # Đọc file đầu vào
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Tính toán số file sẽ được tạo
    total_lines = len(lines)
    file_count = (total_lines + lines_per_file - 1) // lines_per_file  # Làm tròn lên
    
    created_files = []
    
    # Tách và lưu file
    for i in range(file_count):
        start_idx = i * lines_per_file
        end_idx = min((i + 1) * lines_per_file, total_lines)
        
        # Tạo tên file đầu ra
        output_file = os.path.join(output_dir, f"{file_name}_part{i+1}{file_ext}")
        
        # Ghi nội dung vào file đầu ra
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(lines[start_idx:end_idx])
        
        created_files.append(output_file)
        print(f"Đã tạo file: {output_file} với {end_idx - start_idx} dòng")
    
    print(f"Hoàn thành! Đã tách thành {file_count} file.")
    return created_files


split_file('app\data\VN/translated_output.txt','app\data/test\TRANS', 500)