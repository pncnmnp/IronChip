use crossterm::event::{self, Event, KeyCode};
use num::traits::{WrappingAdd, WrappingSub};
use num::{range, CheckedAdd, CheckedSub};
use num_traits::Bounded;
use rand::Rng;
use rodio::{source::SineWave, OutputStream, Sink, Source};
use std::env;
use std::fs::File;
use std::io::{stdout, Read, Stdout, Write};
use std::ops::Add;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::time::interval;

struct Register<T> {
    value: T,
    carry: bool,
}

impl<T> Register<T>
where
    T: Add<Output = T>
        + Copy
        + CheckedAdd
        + WrappingAdd
        + CheckedSub
        + WrappingSub
        + Bounded
        + From<u8>,
{
    fn new(value: T) -> Register<T> {
        Register {
            value,
            carry: false,
        }
    }

    fn add(&mut self, register: T) {
        match self.value.checked_add(&register) {
            Some(sum) => {
                self.value = sum;
                self.carry = false;
            }
            None => {
                self.value = self.value.wrapping_add(&register);
                self.carry = true;
            }
        }
    }

    fn sub(&mut self, register: T) {
        self.carry = true;
        match self.value.checked_sub(&register) {
            Some(result) => self.value = result,
            None => {
                let diff = self.value.wrapping_sub(&register);
                self.carry = false;
                self.value = diff;
            }
        }
    }
}

impl<T: Copy> Copy for Register<T> {}

impl<T: Copy> Clone for Register<T> {
    fn clone(&self) -> Register<T> {
        *self
    }
}

struct DelayTimer {
    timer: u8,
}

struct SoundTimer {
    timer: u8,
}

impl DelayTimer {
    fn new() -> DelayTimer {
        DelayTimer { timer: 0 }
    }

    fn set_timer(&mut self, value: u8) {
        self.timer = value
    }

    fn read_timer(&self) -> u8 {
        self.timer
    }

    #[tokio::main]
    async fn decrement_timer(&mut self) {
        let mut ticker = interval(Duration::from_millis(16));
        loop {
            if self.timer == 0 {
                break;
            }
            ticker.tick().await;
            self.timer -= 1;
        }
    }
}

impl SoundTimer {
    fn new() -> SoundTimer {
        SoundTimer { timer: 0 }
    }

    fn set_timer(&mut self, value: u8) {
        self.timer = value
    }

    fn play(&mut self) {
        if self.timer > 1 {
            thread::spawn(|| {
                let (_stream, stream_handle) = OutputStream::try_default().unwrap();
                let sink = Sink::try_new(&stream_handle).unwrap();

                let source = SineWave::new(440.0)
                    .take_duration(Duration::from_secs_f32(0.1))
                    .amplify(0.20);
                sink.append(source);
                sink.sleep_until_end()
            });
            self.timer = 0
        }
    }
}

fn load_fonts(memory: &mut [u8; 4096]) -> [u8; 4096] {
    // Stores fonts starting from 0 to F
    // It is a convention to put it at 050 - 09F
    let fonts: [u8; 80] = [
        0xF0, 0x90, 0x90, 0x90, 0xF0, 0x20, 0x60, 0x20, 0x20, 0x70, 0xF0, 0x10, 0xF0, 0x80, 0xF0,
        0xF0, 0x10, 0xF0, 0x10, 0xF0, 0x90, 0x90, 0xF0, 0x10, 0x10, 0xF0, 0x80, 0xF0, 0x10, 0xF0,
        0xF0, 0x80, 0xF0, 0x90, 0xF0, 0xF0, 0x10, 0x20, 0x40, 0x40, 0xF0, 0x90, 0xF0, 0x90, 0xF0,
        0xF0, 0x90, 0xF0, 0x10, 0xF0, 0xF0, 0x90, 0xF0, 0x90, 0x90, 0xE0, 0x90, 0xE0, 0x90, 0xE0,
        0xF0, 0x80, 0x80, 0x80, 0xF0, 0xE0, 0x90, 0x90, 0x90, 0xE0, 0xF0, 0x80, 0xF0, 0x80, 0xF0,
        0xF0, 0x80, 0xF0, 0x80, 0x80,
    ];
    let mut start_index = 0x50;
    for &font in fonts.iter() {
        memory[start_index] = font;
        start_index += 1;
    }
    return *memory;
}

fn read_program(filename: &str, memory: &mut [u8; 4096], prog_counter: &usize) {
    let mut program = File::open(filename).expect("Failed to open the file.");

    let mut buffer = Vec::new();
    program
        .read_to_end(&mut buffer)
        .expect("Failed to read the file.");

    let mut offset: usize = 0x0;
    for opcode in buffer.iter() {
        let loc: usize = *prog_counter + offset;
        memory[loc] = *opcode;
        offset += 1;
    }
}

macro_rules! parse_opcode {
    ($opcode: expr, $start: expr, $end: expr) => {
        usize::from_str_radix(&$opcode[$start..$end], 16).unwrap()
    };
    ($opcode:expr, $start: expr) => {
        usize::from_str_radix(&$opcode[$start..], 16).unwrap()
    };
}

fn exec_next_opcode<const HEIGHT: usize, const WIDTH: usize>(
    memory: &mut [u8; 4096],
    prog_counter: &mut usize,
    display: &mut [[u8; HEIGHT]; WIDTH],
    stack: &mut Vec<u16>,
    gen_purp_reg: &mut [Register<u8>; 16],
    index_reg: &mut Register<u16>,
    key: &mut Option<u8>,
    delay_timer: &mut DelayTimer,
    sound_timer: &mut SoundTimer,
) {
    let msb = format!("{:02x}", memory[*prog_counter]);
    let lsb = format!("{:02x}", memory[*prog_counter + 1]);
    let opcode = msb + &lsb;
    // println!("{:?}", opcode);

    if &opcode == "00e0" {
        // 00E0: Clear the screen
        *display = [[0; HEIGHT]; WIDTH];
    } else if &opcode == "00ee" {
        // 00EE: Return from a subroutine
        *prog_counter = stack.pop().unwrap() as usize;
    } else if &opcode[0..1] == "1" {
        // 1NNN: Jump to address NNN
        // We need to subtract by 2,
        // else it will jump to the next instruction after NNN
        *prog_counter = parse_opcode!(opcode, 1) - 2;
    } else if &opcode[0..1] == "2" {
        // 2NNN: Execute subroutine starting at address NNN
        stack.push(*prog_counter as u16);
        *prog_counter = parse_opcode!(opcode, 1) - 2;
    } else if &opcode[0..1] == "3" {
        // 3XNN: Skip the following instruction if
        //       the value of register VX equals NN
        let x = parse_opcode!(opcode, 1, 2);
        let nn = parse_opcode!(opcode, 1) as u8;
        if gen_purp_reg[x].value == nn {
            *prog_counter += 2;
        }
    } else if &opcode[0..1] == "4" {
        // 4XNN: Skip the following instruction if
        //       the value of register VX is not equal to NN
        let x = parse_opcode!(opcode, 1, 2);
        let nn = parse_opcode!(opcode, 2) as u8;
        if gen_purp_reg[x].value != nn {
            *prog_counter += 2;
        }
    } else if &opcode[0..1] == "5" {
        // 5XY0: Skip the following instruction if
        //       the value of register VX is equal to the value of register VY
        let x = parse_opcode!(opcode, 1, 2);
        let y = parse_opcode!(opcode, 2, 3);
        if gen_purp_reg[x].value == gen_purp_reg[y].value {
            *prog_counter += 2;
        }
    } else if &opcode[0..1] == "6" {
        // 6XNN: Store number NN in register VX
        let x = parse_opcode!(opcode, 1, 2);
        let nn = parse_opcode!(opcode, 2) as u8;
        gen_purp_reg[x].value = nn;
    } else if &opcode[0..1] == "7" {
        // 7XNN: Add the value NN to register VX
        let x = parse_opcode!(opcode, 1, 2);
        let nn = parse_opcode!(opcode, 2) as u8;
        gen_purp_reg[x].add(nn);
    } else if &opcode[0..1] == "8" {
        if &opcode[3..4] == "0" {
            // 8XY0: Store the value of register VY in register VX
            let x = parse_opcode!(opcode, 1, 2);
            let y = parse_opcode!(opcode, 2, 3);
            gen_purp_reg[x].value = gen_purp_reg[y].value;
        } else if &opcode[3..4] == "1" {
            // 8XY1: Set VX to VX OR VY
            let x = parse_opcode!(opcode, 1, 2);
            let y = parse_opcode!(opcode, 2, 3);
            gen_purp_reg[x].value = gen_purp_reg[x].value | gen_purp_reg[y].value;
            gen_purp_reg[0xF].value = 0;
        } else if &opcode[3..4] == "2" {
            // 8XY2: Set VX to VX AND VY
            let x = parse_opcode!(opcode, 1, 2);
            let y = parse_opcode!(opcode, 2, 3);
            gen_purp_reg[x].value = gen_purp_reg[x].value & gen_purp_reg[y].value;
            gen_purp_reg[0xF].value = 0;
        } else if &opcode[3..4] == "3" {
            // 8XY3: Set VX to VX XOR VY
            let x = parse_opcode!(opcode, 1, 2);
            let y = parse_opcode!(opcode, 2, 3);
            gen_purp_reg[x].value = gen_purp_reg[x].value ^ gen_purp_reg[y].value;
            gen_purp_reg[0xF].value = 0;
        } else if &opcode[3..4] == "4" {
            // 8XY4: Add the value of register VY to register VX
            //       Set VF to 01 if a carry occurs
            //       Set VF to 00 if a carry does not occur
            let x = parse_opcode!(opcode, 1, 2);
            let y = parse_opcode!(opcode, 2, 3);

            gen_purp_reg[x].add(gen_purp_reg[y].value);
            let carry = gen_purp_reg[x].carry as u8;
            gen_purp_reg[0xF].value = carry;
        } else if &opcode[3..4] == "5" {
            // 8XY5: Subtract the value of register VY from register VX
            //       Set VF to 00 if a borrow occurs
            //       Set VF to 01 if a borrow does not occur
            let x = parse_opcode!(opcode, 1, 2);
            let y = parse_opcode!(opcode, 2, 3);

            gen_purp_reg[x].sub(gen_purp_reg[y].value);
            let borrow = gen_purp_reg[x].carry as u8;
            gen_purp_reg[0xF].value = borrow;
        } else if &opcode[3..4] == "6" {
            // 8XY6: Store the value of register VY shifted right one bit in register VX
            //       Set register VF to the least significant bit prior to the shift
            //       VY is unchanged
            let x = parse_opcode!(opcode, 1, 2);

            // Starting with CHIP-48 and SUPER-CHIP in the early 1990s,
            // these instructions were changed so that they shifted VX in place,
            // and ignored the Y completely.
            let lsb = gen_purp_reg[x].value & 1;
            gen_purp_reg[x].value = gen_purp_reg[x].value >> 1;
            gen_purp_reg[0xF].value = lsb;
        } else if &opcode[3..4] == "7" {
            // 8XY7: Set register VX to the value of VY minus VX
            //       Set VF to 00 if a borrow occurs
            //       Set VF to 01 if a borrow does not occur
            let x = parse_opcode!(opcode, 1, 2);
            let y = parse_opcode!(opcode, 2, 3);

            let reg_y_val = gen_purp_reg[y].value;

            gen_purp_reg[y].sub(gen_purp_reg[x].value);
            gen_purp_reg[x].value = gen_purp_reg[y].value;
            let borrow = gen_purp_reg[y].carry as u8;
            gen_purp_reg[0xF].value = borrow;

            // Borrow does not need to be updated back
            gen_purp_reg[y].value = reg_y_val;
        } else if &opcode[3..4] == "e" {
            // 8XYE: Store the value of register VY shifted left one bit in register VX
            //       Set register VF to the most significant bit prior to the shift
            //       VY is unchanged
            let x = parse_opcode!(opcode, 1, 2);
            let msb = (gen_purp_reg[x].value >> 7) & 1;
            gen_purp_reg[x].value = gen_purp_reg[x].value << 1;
            gen_purp_reg[0xF].value = msb;
        }
    } else if &opcode[0..1] == "9" {
        // 9XY0: Skip the following instruction if
        //       the value of register VX is not equal to the value of register VY
        let x = parse_opcode!(opcode, 1, 2);
        let y = parse_opcode!(opcode, 2, 3);
        if gen_purp_reg[x].value != gen_purp_reg[y].value {
            *prog_counter += 2;
        }
    } else if &opcode[0..1] == "a" {
        // ANNN: Store memory address NNN in register I
        let addr = parse_opcode!(opcode, 1) as u16;
        index_reg.value = addr;
    } else if &opcode[0..1] == "b" {
        // BNNN: Jump to address NNN + V0
        let addr = parse_opcode!(opcode, 1);
        *prog_counter = (gen_purp_reg[0].value as usize) + addr;
    } else if &opcode[0..1] == "c" {
        // CNNN: Set VX to a random number with a mask of NN
        let x = parse_opcode!(opcode, 1, 2);
        let nn = parse_opcode!(opcode, 2) as u8;
        let mut rng = rand::thread_rng();
        let rand_num: u8 = rng.gen_range(0..=255);
        gen_purp_reg[x].value = rand_num & nn;
    } else if &opcode[0..1] == "d" {
        // DXYN: Draw a sprite at position VX, VY with
        //       N bytes of sprite data starting at the address stored in I
        //       Set VF to 01 if any set pixels are changed to unset,
        //          and 00 otherwise
        let x = parse_opcode!(opcode, 1, 2);
        let vx = gen_purp_reg[x].value as usize;

        let y = parse_opcode!(opcode, 2, 3);
        let vy = gen_purp_reg[y].value as usize;

        let n = parse_opcode!(opcode, 3, 4);

        let addr = index_reg.value as usize;
        let sprite = &memory[addr..addr + n];
        let mut collision = false;

        for i in range(0, sprite.len()) {
            for j in range(0, 8) {
                let bit = (sprite[i] >> (7 - j)) & 1;
                if (vy + i < HEIGHT) & (vx + j < WIDTH) {
                    if (display[vx + j][vy + i] == 1) & (bit == 1) {
                        collision = true;
                    }
                    display[vx + j][vy + i] = display[vx + j][vy + i] ^ bit;
                }
            }
        }
        match collision {
            true => gen_purp_reg[0xF].value = 1,
            false => gen_purp_reg[0xF].value = 0,
        }
    } else if &opcode[0..1] == "e" {
        key_handler(key);
        if &opcode[2..] == "9e" {
            // EX9E: Skip the following instruction if
            //       the key corresponding to the hex value
            //       currently stored in register VX is pressed
            let x = parse_opcode!(opcode, 1, 2);
            match key {
                Some(_) => {
                    if key.unwrap() == gen_purp_reg[x].value {
                        *prog_counter += 2;
                    }
                }
                None => (),
            }
        } else if &opcode[2..] == "a1" {
            // EXA1: Skip the following instruction if
            //       the key corresponding to the hex value
            //       currently stored in register VX is not pressed
            let x = parse_opcode!(opcode, 1, 2);
            match key {
                Some(_) => {
                    if key.unwrap() != gen_purp_reg[x].value {
                        *prog_counter += 2;
                    }
                }
                None => (),
            }
        }
    } else if &opcode[0..1] == "f" {
        if &opcode[2..] == "07" {
            // FX07: Store the current value of the delay timer in register VX
            let x = parse_opcode!(opcode, 1, 2);
            gen_purp_reg[x].value = delay_timer.read_timer();
        } else if &opcode[2..] == "0a" {
            // FX0A: Wait for a keypress and store the result in register VX
            loop {
                key_handler(key);
                match key {
                    None => (),
                    Some(_) => break,
                }
            }
            let x = parse_opcode!(opcode, 1, 2);
            gen_purp_reg[x].value = key.unwrap();
        } else if &opcode[2..] == "15" {
            // FX15: Set the delay timer to the value of register VX
            let x = parse_opcode!(opcode, 1, 2);
            delay_timer.set_timer(gen_purp_reg[x].value);
            delay_timer.decrement_timer();
        } else if &opcode[2..] == "18" {
            // FX18: Set the sound timer to the value of register VX
            let x = parse_opcode!(opcode, 1, 2);
            sound_timer.set_timer(gen_purp_reg[x].value);
            sound_timer.play();
        } else if &opcode[2..] == "1e" {
            // FX1E: Add the value stored in register VX to register I
            let x = parse_opcode!(opcode, 1, 2);
            index_reg.value += gen_purp_reg[x].value as u16;
        } else if &opcode[2..] == "29" {
            // FX29: Set I to the memory address of the sprite data
            //       corresponding to the hexadecimal digit stored in register VX
            let x = parse_opcode!(opcode, 1, 2);
            let digit = gen_purp_reg[x].value as u16;
            index_reg.value = 0x50 + (digit * 5);
        } else if &opcode[2..] == "33" {
            // FX33: Store the binary-coded decimal equivalent of
            //       the value stored in register VX at addresses I, I + 1, and I + 2
            let x = parse_opcode!(opcode, 1, 2);
            let value = gen_purp_reg[x].value;

            let leftmost = value / 10 / 10;
            let middle = (value / 10) - (leftmost * 10);
            let rightmost = value - (leftmost * 100) - (middle * 10);

            memory[index_reg.value as usize] = leftmost;
            memory[(index_reg.value as usize) + 1] = middle;
            memory[(index_reg.value as usize) + 2] = rightmost;
        } else if &opcode[2..] == "55" {
            // FX55: Store the values of registers V0 to VX inclusive in memory starting at address I
            //       I is set to I + X + 1 after operation
            let x = parse_opcode!(opcode, 1, 2);
            for index in range(0, x + 1) {
                memory[(index_reg.value as usize) + index] = gen_purp_reg[index].value;
            }
        } else if &opcode[2..] == "65" {
            // FX65: Fill registers V0 to VX inclusive with the values stored in memory starting at address I
            //       I is set to I + X + 1 after operation
            let x = parse_opcode!(opcode, 1, 2);
            for index in range(0, x + 1) {
                gen_purp_reg[index].value = memory[(index_reg.value as usize) + index];
            }
        }
    }
}

fn main() {
    let mut memory: [u8; 4096] = [0x0; 4096];
    load_fonts(&mut memory);

    const DISP_WIDTH: usize = 64;
    const DISP_HEIGHT: usize = 32;
    let mut display: [[u8; DISP_HEIGHT]; DISP_WIDTH] = [[0; DISP_HEIGHT]; DISP_WIDTH];

    // To call subroutines/functions and return from them
    let mut stack: Vec<u16> = Vec::new();

    // Point at locations in memory
    let mut index_reg: Register<u16> = Register::new(0x0);
    // From V0 to VF
    // VF acts as a flag register
    let mut gen_purp_reg: [Register<u8>; 16] = [Register::new(0x0); 16];

    // Points at current instruction in memory
    // Need 12 bits for this => 2^12 = 4096
    let mut prog_counter: usize = 0x200;

    let mut delay_timer: DelayTimer = DelayTimer::new();
    let mut sound_timer: SoundTimer = SoundTimer::new();

    let program_file: Vec<String> = env::args().collect();
    if program_file.len() >= 2 {
        read_program(&program_file[1].as_str(), &mut memory, &prog_counter);
    } else {
        panic!("File name not passed as argument.");
    }
    // println!("{:x?}", &memory[0x200..]);

    let cycles_per_second: u128 = 700;
    let cycle_duration: Duration = Duration::from_nanos(1_000_000_000 / cycles_per_second as u64);

    let cycle_counter: Arc<Mutex<u128>> = Arc::new(Mutex::new(0));
    let start_time: Instant = Instant::now();

    let counter_clone: Arc<Mutex<u128>> = Arc::clone(&cycle_counter);
    let _handle: thread::JoinHandle<_> = thread::spawn(move || loop {
        let elapsed_time: Duration = start_time.elapsed();
        let elapsed_cycles: u128 = elapsed_time.as_micros() * cycles_per_second / 1_000_000;

        {
            let mut counter: std::sync::MutexGuard<u128> = counter_clone.lock().unwrap();
            if elapsed_cycles > *counter {
                *counter = elapsed_cycles;
            }
        }

        thread::sleep(cycle_duration);
    });

    let iter_threshold: u128 = 10;
    let mut key: Option<u8> = None;

    let mut stdout = stdout();
    let _raw_mode = crossterm::terminal::enable_raw_mode().unwrap();
    // Clears the screen
    // From: https://stackoverflow.com/a/34837038/7543474
    print!("{}[2J", 27 as char);

    loop {
        let counter_value: u128 = *cycle_counter.lock().unwrap();
        if counter_value > iter_threshold {
            exec_next_opcode(
                &mut memory,
                &mut prog_counter,
                &mut display,
                &mut stack,
                &mut gen_purp_reg,
                &mut index_reg,
                &mut key,
                &mut delay_timer,
                &mut sound_timer,
            );
            prog_counter += 2;

            // println!("{:?}", key);

            print_display(&display, &mut stdout);

            // Reset the cycle counter
            *cycle_counter.lock().unwrap() = 0;
        }
    }
}

fn key_handler(key: &mut Option<u8>) {
    // Mapping from QWERTY to CHIP-8 format
    // CHIP-8        QWERTY
    // 1 2 3 C       1 2 3 4
    // 4 5 6 D       Q W E R
    // 7 8 9 E       A S D F
    // A 0 B F       Z X C V
    if event::poll(Duration::from_millis(10)).unwrap() {
        if let Event::Key(key_event) = event::read().unwrap() {
            match key_event.code {
                KeyCode::Char('1') => *key = Some(1),
                KeyCode::Char('2') => *key = Some(2),
                KeyCode::Char('3') => *key = Some(3),
                KeyCode::Char('4') => *key = Some(0xC),
                KeyCode::Char('q') => *key = Some(4),
                KeyCode::Char('w') => *key = Some(5),
                KeyCode::Char('e') => *key = Some(6),
                KeyCode::Char('r') => *key = Some(0xD),
                KeyCode::Char('a') => *key = Some(7),
                KeyCode::Char('s') => *key = Some(8),
                KeyCode::Char('d') => *key = Some(9),
                KeyCode::Char('f') => *key = Some(0xE),
                KeyCode::Char('z') => *key = Some(0xA),
                KeyCode::Char('x') => *key = Some(0),
                KeyCode::Char('c') => *key = Some(0xB),
                KeyCode::Char('v') => *key = Some(0xF),
                _ => *key = None,
            }
            return;
        }
    }
}

fn print_display<const HEIGHT: usize, const WIDTH: usize>(
    display: &[[u8; HEIGHT]; WIDTH],
    stdout: &mut Stdout,
) {
    // Move the cursor to the beginning of the terminal
    stdout.write_all(b"\x1B[1;1H").unwrap();

    for i in 0..HEIGHT {
        for j in 0..WIDTH {
            let pixel = if display[j][i] == 1 { "\u{2588}" } else { " " };
            stdout.write_all(pixel.as_bytes()).unwrap();
        }
        stdout.write_all(b"\r\n").unwrap();
    }

    // Flush the output buffer to ensure that the output is immediately displayed
    stdout.flush().unwrap();
}
