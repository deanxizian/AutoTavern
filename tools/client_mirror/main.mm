// Loads the user's installed, HearthSim-signed macOS reader. No vendor binary
// is copied into this project. Exposes lobby, scene and local-player metadata only.
#import <AppKit/AppKit.h>
#import <Security/Security.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <mach/mach.h>
#include <unistd.h>
#include <exception>

@protocol ATMirror
- (instancetype)initWithPID:(pid_t)pid withBlocking:(BOOL)blocking;
- (NSArray<NSNumber *> *)getAvailableBattlegroundsRaces;
- (NSString *)getLogSessionDir;
- (id)getGameServerInfo;
- (NSNumber *)getGameType;
- (id)getSceneMgrState;
- (id)getMatchInfo;
@end

static NSDictionary *publicState(id<ATMirror> mirror) {
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    id scene = [mirror getSceneMgrState];
    if (scene) result[@"scene"] = @{@"mode": [scene valueForKey:@"mode"] ?: @0,
        @"loaded": [scene valueForKey:@"sceneLoaded"] ?: @NO,
        @"transitioning": [scene valueForKey:@"transitioning"] ?: @YES};
    id match = [mirror getMatchInfo];
    if (match) {
        id pid = [[match valueForKey:@"localPlayer"] valueForKey:@"playerId"];
        if ([pid isKindOfClass:[NSNumber class]]) result[@"local_player_id"] = pid;
        result[@"spectator"] = [match valueForKey:@"spectator"] ?: @YES;
    }
    // Card entities, zone order and choices have one source: Power.log.
    return result;
}

static int outputFD;
static void emit(NSDictionary *value) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:value options:0 error:nil];
    if (!data) return;
    const char *bytes = (const char *)data.bytes;
    size_t remaining = data.length;
    while (remaining) {
        ssize_t n = write(outputFD, bytes, remaining);
        if (n <= 0) exit(0);
        remaining -= n; bytes += n;
    }
    if (write(outputFD, "\n", 1) != 1) exit(0);
}

static NSString *loadMirror(Class *mirrorClass, NSString **frameworkPath) {
    NSArray *apps = @[@"/Applications/HSTracker.app",
                     [NSHomeDirectory() stringByAppendingPathComponent:@"Applications/HSTracker.app"]];
    for (NSString *app in apps) {
        NSString *framework = [app stringByAppendingPathComponent:@"Contents/Frameworks/HearthMirror.framework"];
        if (![[NSFileManager defaultManager] fileExistsAtPath:framework]) continue;
        SecStaticCodeRef code = nullptr;
        SecRequirementRef requirement = nullptr;
        OSStatus status = SecStaticCodeCreateWithPath((__bridge CFURLRef)[NSURL fileURLWithPath:framework], kSecCSDefaultFlags, &code);
        if (status == errSecSuccess) status = SecRequirementCreateWithString(
            CFSTR("anchor apple generic and identifier \"com.ifehervari.HearthMirror\" and certificate leaf[subject.OU] = \"RL7C49LAMC\""),
            kSecCSDefaultFlags, &requirement);
        if (status == errSecSuccess) status = SecStaticCodeCheckValidity(code, kSecCSCheckAllArchitectures, requirement);
        if (requirement) CFRelease(requirement);
        if (code) CFRelease(code);
        if (status != errSecSuccess) return @"framework_signature_invalid";
        NSString *binary = [framework stringByAppendingPathComponent:@"Versions/A/HearthMirror"];
        if (!dlopen(binary.fileSystemRepresentation, RTLD_NOW | RTLD_LOCAL)) return @"framework_load_failed";
        Class cls = NSClassFromString(@"HearthMirror");
        for (NSString *method in @[@"initWithPID:withBlocking:", @"getAvailableBattlegroundsRaces",
                                   @"getLogSessionDir", @"getGameServerInfo", @"getGameType"]) {
            if (!cls || !class_getInstanceMethod(cls, NSSelectorFromString(method))) return @"framework_api_incompatible";
        }
        *mirrorClass = cls; *frameworkPath = framework;
        return nil;
    }
    return @"framework_missing";
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        // Keep framework diagnostics (potentially containing unrelated client
        // values) away from our bounded JSON protocol and persistent logs.
        outputFD = dup(STDOUT_FILENO);
        freopen("/dev/null", "w", stdout);
        freopen("/dev/null", "w", stderr);
        Class mirrorClass = Nil;
        NSString *frameworkPath = nil;
        NSString *loadError = loadMirror(&mirrorClass, &frameworkPath);
        if (argc == 2 && strcmp(argv[1], "--probe") == 0) {
            emit(@{@"schema": @"autotavern.races.v1", @"status": loadError ?: @"component_ready",
                   @"framework": frameworkPath ?: @"", @"provider": @"hearthmirror"});
            return loadError ? 1 : 0;
        }
        id<ATMirror> mirror = nil;
        pid_t mirrorPID = 0;
        NSString *mirrorKey = nil;
        char *line = nullptr; size_t capacity = 0;
        while (getline(&line, &capacity, stdin) > 0) {
            @autoreleasepool {
                if (strlen(line) > 4096) break;
                NSDictionary *request = [NSJSONSerialization JSONObjectWithData:[[NSString stringWithUTF8String:line] dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
                if (![request isKindOfClass:[NSDictionary class]] || ![request[@"game_key"] isKindOfClass:[NSString class]]) break;
                BOOL wantPublic = [request[@"scope"] isEqual:@"public_state"];
                NSMutableDictionary *result = [@{@"schema": wantPublic ? @"autotavern.memory.v1" : @"autotavern.races.v1", @"provider": @"hearthmirror",
                    @"game_key": request[@"game_key"], @"status": @"memory_not_ready"} mutableCopy];
                @try {
                    try {
                        do {
                            if (loadError) { result[@"status"] = loadError; break; }
                            NSArray<NSRunningApplication *> *games = [NSRunningApplication runningApplicationsWithBundleIdentifier:@"unity.Blizzard Entertainment.Hearthstone"];
                            if (games.count != 1) { mirror = nil; mirrorPID = 0;
                                result[@"status"] = games.count ? @"multiple_clients" : @"client_not_running"; break; }
                            NSRunningApplication *game = games.firstObject;
                            pid_t pid = game.processIdentifier;
                            result[@"pid"] = @(pid);
                            if (!mirror || mirrorPID != pid || ![mirrorKey isEqual:request[@"game_key"]]) {
                                mirror = nil;
                                mach_port_t task = MACH_PORT_NULL;
                                kern_return_t access = task_for_pid(mach_task_self(), pid, &task);
                                if (access != KERN_SUCCESS) { result[@"status"] = @"memory_access_denied"; result[@"mach_error"] = @(access); break; }
                                mach_port_deallocate(mach_task_self(), task);
                                mirror = [(id<ATMirror>)[mirrorClass alloc] initWithPID:pid withBlocking:NO];
                                mirrorPID = pid; mirrorKey = request[@"game_key"];
                            }
                            result[@"memory_access"] = @YES;
                            // Read only the handle from the server object; never serialize
                            // addresses, connection passwords or accounts.
                            NSNumber *before = [[mirror getGameServerInfo] valueForKey:@"gameHandle"];
                            NSArray *races = wantPublic ? nil : [mirror getAvailableBattlegroundsRaces];
                            NSString *session = [mirror getLogSessionDir];
                            NSNumber *gameType = [mirror getGameType];
                            if (wantPublic) {
                                BOOL compatible = YES;
                                for (NSString *method in @[@"getSceneMgrState", @"getMatchInfo"]) {
                                    if (!class_getInstanceMethod(mirrorClass, NSSelectorFromString(method))) compatible = NO;
                                }
                                if (!compatible) { result[@"status"] = @"framework_api_incompatible"; break; }
                            }
                            NSDictionary *visible = wantPublic ? publicState(mirror) : nil;
                            NSNumber *after = [[mirror getGameServerInfo] valueForKey:@"gameHandle"];
                            if ([session isKindOfClass:[NSString class]]) result[@"log_session"] = session.lastPathComponent;
                            if ([gameType isKindOfClass:[NSNumber class]]) result[@"game_type"] = gameType;
                            if (wantPublic) {
                                if (game.terminated || ![session isKindOfClass:[NSString class]] ||
                                    !((!before && !after) || [before isEqual:after])) break;
                                result[@"status"] = @"public_read";
                                result[@"public_state"] = visible;
                                result[@"game_handle"] = after ?: @0;
                                break;
                            }
                            if (game.terminated || !before || ![before isEqual:after] || before.longLongValue <= 0) break;
                            if (![races isKindOfClass:[NSArray class]] || races.count > 16 || ![session isKindOfClass:[NSString class]]) break;
                            BOOL numbers = YES;
                            for (id race in races) if (![race isKindOfClass:[NSNumber class]]) numbers = NO;
                            if (!numbers) { result[@"status"] = @"invalid_memory_response"; break; }
                            result[@"status"] = @"read";
                            result[@"race_ids"] = races;
                            result[@"game_handle"] = after;
                            result[@"game_type"] = gameType ?: @0;
                        } while (false);
                    } catch (const std::exception &) { result[@"status"] = @"memory_read_failed"; mirror = nil; }
                      catch (...) { result[@"status"] = @"memory_read_failed"; mirror = nil; }
                } @catch (NSException *) { result[@"status"] = @"memory_read_failed"; mirror = nil; }
                emit(result);
            }
        }
        free(line);
    }
    return 0;
}
