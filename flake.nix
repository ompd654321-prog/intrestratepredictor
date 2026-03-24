{
  description = "Interest Rate Predictor Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      # Support for Linux and macOS (Intel and Apple Silicon)
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          # Define the Python environment with required dependencies
          pythonEnv = pkgs.python3.withPackages (ps: with ps; [
            flask
            scikit-learn
            pandas
            numpy
            joblib
          ]);
        in {
          default = pkgs.mkShell {
            packages = [ pythonEnv ];

            shellHook = ''
              echo "✅ Interest Rate Predictor environment loaded!"
              echo "🚀 Run the app using: python app.py"
            '';
          };
        });
    };
}